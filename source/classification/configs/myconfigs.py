import argparse


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "project_path",
        default='/mnt/DATA/research/project/classificationCOVID19applyseg/',
        type=str)

    parser.add_argument(
        "--checkpoint_path",
        default='result/classification/model/', type=str)

    parser.add_argument(
        '--img_path',
        default='dataset/COVIDxCXR3/', type=str)

    parser.add_argument(
        '--mask_path',
        default='result/segmentation/lung_mask/', type=str)

    parser.add_argument(
        "--result_path",
        default='result/classification/report/', type=str)

    parser.add_argument('--batch_size', default=8, type=int)

    parser.add_argument('--num_epochs', default=70, type=int)
    # parser.add_argument('--num_epochs', default= 1, type=int)

    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--classes', default=['Negative', 'Positive'])
    parser.add_argument('--img_size', default=256, type=int)

    parser.add_argument('--mean', default=[0.5002, 0.5002, 0.5003])
    parser.add_argument('--std', default=[0.2079, 0.2079, 0.2079])

    parser.add_argument(
        '--model_name',
        default=[
            'ResNet152', 'VGG19_bn', 'ResNet18', 'ResNet50',
            'ResNet101', 'EfficientNetB7', 'InceptionResNetV2'])

    parser.add_argument(
        '--feature_extract', action='store_true', default=False)

    opt = parser.parse_args()
    return opt
