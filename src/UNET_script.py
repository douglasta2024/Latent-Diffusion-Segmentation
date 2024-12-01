import os
import torch
from monai.networks.nets import UNet
from monai.data import ArrayDataset, DataLoader
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.transforms import (
    Compose,
    RandSpatialCrop,
    LoadImage,
    EnsureChannelFirst,
    ScaleIntensityRange,
    Orientation,
)
from ignite.engine import Events, Engine
from ignite.metrics import Recall, Precision
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import boto3
from io import BytesIO
import nibabel as nb
from torch import from_numpy
import tempfile

def generate_output():
    ### GLOBAL VARIABLES
    # ROOT_PATH = os.path.join(os.getcwd(), "main", "src")
    # DATA_PATH = os.path.join(ROOT_PATH, "data")
    DEVICE = torch.device("cpu")    

    # defines basemodel
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(DEVICE)    
    print("Base Model Loaded In")

    # connection to S3 database
    s3 = boto3.client(
        service_name='s3',
        region_name='us-east-2',
        aws_access_key_id='AKIAUQ4L3FBTESMORT75',
        aws_secret_access_key='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',
    )
    bucket_name = "ct-scans--use2-az1--x-s3"
    imgs = []
    segs = []
    response = s3.list_objects_v2(Bucket=bucket_name)
    if 'Contents' in response:
        for obj in response['Contents']:
            file_name = obj['Key']
            
            #loads in model
            if "model" in file_name:
                single_response = s3.get_object(Bucket=bucket_name, Key=file_name)
                model_bytes = BytesIO(single_response['Body'].read())
                model_weights = torch.load(model_bytes, weights_only=True, map_location=DEVICE) 

                # loading model weights onto model
                model.load_state_dict(model_weights) 
                print("Model Successfully Loaded In")
                continue

            single_response = s3.get_object(Bucket=bucket_name, Key=file_name)
            nifti_bytes = single_response['Body'].read()

            # Create a BytesIO object from the bytes
            nifti_file = BytesIO(nifti_bytes)

            # Use FileHolder and from_file_map to load the NIfTI file
            file_map = nb.FileHolder(fileobj=nifti_file)
            nifti_image = nb.Nifti1Image.from_file_map({'header': file_map, 'image': file_map})
            img = nifti_image.get_fdata()
            img = from_numpy(img)
            if "volume" in file_name:
                img = img.unsqueeze(dim=0)
                imgs.append(img)
            else:
                img = img.unsqueeze(dim=0)
                segs.append(img)

        print("Images ready for testing.")

    # evaluation setup
    loss = DiceLoss()
    store_predictions = []
    mean_dice_metric = DiceMetric(include_background=True, reduction="mean")
    mean_dice_score = -1

    def evaluation_step(engine, batch):
        model.eval()
        with torch.no_grad():
            images, masks = batch
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            outputs = model(images)

            logits = model(images)

            if logits.shape[1] == 1:  # Binary segmentation
                predictions = torch.sigmoid(logits) > 0.5  # Threshold at 0.5
            else:  # Multi-class segmentation
                predictions = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            
            return predictions, masks        

    evaluator = Engine(evaluation_step)

    # Attach DiceMetric calculation at every iteration
    @evaluator.on(Events.ITERATION_COMPLETED)
    def update_dice_metric(engine):
        predictions, masks = engine.state.output
        mean_dice_metric(y_pred=predictions, y=masks)

    # Log results at the end of evaluation
    @evaluator.on(Events.COMPLETED)
    def log_mean_dice(engine):
        mean_dice_score = mean_dice_metric.aggregate().item()
        print(f"Mean Dice Score: {mean_dice_score:.4f}")
        #mean_dice_metric.reset()  # Reset metric for next run

    Recall(average=True).attach(evaluator, "recall")
    Precision(average=False).attach(evaluator, "precision")

    @evaluator.on(Events.ITERATION_COMPLETED)
    def save_segmentation_masks(engine):
        predictions, ground_truth = engine.state.output  # Get predictions and masks from the step function
        
        # Save predicted masks
        for idx, prediction in enumerate(predictions):
            store_predictions.append([prediction*1, ground_truth[idx]])

    amin = -22.18
    amax = 450.0

    image_transforms = Compose(
        [
            #LoadImage(image_only=True, ensure_channel_first=True),\
            ScaleIntensityRange(
                a_min=amin,
                a_max=amax,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            Orientation(axcodes="RAS"),   
            RandSpatialCrop(
            (512,512,160), 
                random_center=False
            ), 
        ]
    )

    seg_transforms = Compose(
        [
            #LoadImage(image_only=True, ensure_channel_first=True),
            Orientation(axcodes="RAS"),
            RandSpatialCrop(
            (512,512,160), 
                random_center=False
            ), 
        ]
    )

    test_ds = ArrayDataset(img=imgs, img_transform=image_transforms, seg=segs, seg_transform=seg_transforms)
    test_loader = DataLoader(test_ds, batch_size=1, pin_memory=torch.cuda.is_available())

    print("Initiating testing...")
    evaluator.run(test_loader)

    mean_dice_score = mean_dice_metric.aggregate().item()
    recall = evaluator.state.metrics["recall"]
    precision = evaluator.state.metrics["precision"]
    print(f"Recall: {recall} | Precision {precision}")

    print("Starting GIF Generation")    
    mask = store_predictions[0][0].squeeze().T
    ground_truth = store_predictions[0][1].squeeze().T
    mask2 = store_predictions[1][0].squeeze().T
    ground_truth2 = store_predictions[1][1].squeeze().T

    def gif_generator(image, idx):
        fig, ax = plt.subplots()
        image_slice = ax.imshow(image[0].numpy(), cmap="bone", animated=True)

        # update function for each frame
        def update(frame):
            image_slice.set_array(image[frame].numpy())  # Update the image data
            ax.set_title(f"Slice {frame}")
            return [image_slice]

        # Create an animation
        ani = FuncAnimation(fig, update, frames=image.shape[0], interval=300, blit=True)    
        
        # saving gif to bucket
        # file_name = f"mask{idx}.gif"
        # s3.upload_fileobj(ani, bucket_name, file_name)
        # print(f"Uploaded {file_name}")
        return ani

    print("Initiating GIF Generation")
    masks = [mask, ground_truth, mask2, ground_truth2]
    gifs = []
    for idx, img in enumerate(masks):
        ani = gif_generator(img, idx)
        
        # saving animation to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".gif") as temp_file:
            writer = animation.PillowWriter()
            ani.save(temp_file.name, writer=writer)

            # reading file into byteio
            temp_file.seek(0)
            gif_buffer = BytesIO(temp_file.read())
            #gif = base64.b64encode(gif_buffer().decode("ascii"))
            gifs.append(gif_buffer)
        os.remove(temp_file.name)
        print(f"Mask {idx} Initialized")
        #ani.save(os.path.join(ROOT_PATH, f"saved_gifs/mask{idx}.gif"), writer="pillow")


    return mean_dice_score, recall, precision, gifs


