import argparse


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--end_epoch', default=50, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument(
        '--project_path',
        default='/mnt/DATA/research/project/classificationCOVID19applyseg/',
        type=str)

    parser.add_argument(
        '--checkpoint_path',
        default='result/segmentation/model/checkpointFPNDenseNet121.pt',
        type=str)

    parser.add_argument(
        '--model_path',
        default='result/segmentation/model/modelFPNDenseNet121.pt', type=str)

    parser.add_argument(
        '--data_path',
        default='dataset/COVID_QU_Ex/' +
        'Lung Segmentation Data/Lung Segmentation Data/',
        type=str)

    parser.add_argument('--first_train', default=1, type=int)
    parser.add_argument('--device', default=True, type=bool)
    parser.add_argument('--mean', default=[0.5128])
    parser.add_argument('--std', default=[0.2220])
    parser.add_argument('--img_size', default=256, type=int)
    opt = parser.parse_args()
    return opt
