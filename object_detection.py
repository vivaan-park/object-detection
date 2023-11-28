import os
import torch
import torchvision

from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import v2 as T

import utils
from engine import train_one_epoch, evaluate

import ssl

ssl._create_default_https_context = ssl._create_unverified_context


class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # 이미지 정렬
        self.imgs = list(sorted(os.listdir(os.path.join(root, 'PNGImages'))))
        self.masks = list(sorted(os.listdir(os.path.join(root, 'PedMasks'))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, 'PNGImages', self.imgs[idx])
        mask_path = os.path.join(self.root, 'PedMasks', self.masks[idx])
        img = read_image(img_path)
        mask = read_image(mask_path)
        obj_ids = torch.unique(mask)
        obj_ids = obj_ids[1:]  # obj_ids[0]은 배경이므로 제거
        num_objs = len(obj_ids)

        # 색상으로 인코딩된 마스크를 바이너리 마스크로 변환
        masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)

        # 각 마스크에 대한 경계상자
        boxes = masks_to_boxes(masks)

        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = idx
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # 샘플과 타겟을 tv_tensors로 래핑
        img = tv_tensors.Image(img)

        target = {}
        target['boxes'] = tv_tensors.BoundingBoxes(boxes, format='XYXY', canvas_size=F.get_size(img))
        target['masks'] = tv_tensors.Mask(masks)
        target['labels'] = labels
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_model_instance_segmentation(num_classes):
    # pre-trained 모델 로드
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT')

    # featrue 수
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # 헤드 교체
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # mask feature 수
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # mask 예측 교체
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model


def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)


def main():
    device = (
        'cuda'
        if torch.cuda.is_available()
        else 'mps'
        if torch.backends.mps.is_available()
        else 'cpu'
    )

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    dataset = PennFudanDataset('data/PennFudanPed', get_transform(train=True))
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        collate_fn=utils.collate_fn
    )

    images, targets = next(iter(data_loader))
    images = list(image for image in images)
    targets = [{k: v for k, v in t.items()} for t in targets]
    output = model(images, targets)
    print(output)

    model.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    predictions = model(x)
    print(predictions[0])

    num_classes = 2 # 배경 + 사람
    dataset = PennFudanDataset('data/PennFudanPed', get_transform(train=True))
    dataset_test = PennFudanDataset('data/PennFudanPed', get_transform(train=False))

    # train / test 분리
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # train / validation 정의
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        collate_fn=utils.collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=utils.collate_fn
    )

    model = get_model_instance_segmentation(num_classes)

    model.to(device)

    # optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )

    # lr 스케쥴러
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )

    num_epochs = 2

    for epoch in range(num_epochs):
        # 1 에폭동안 학습하고 10회마다 출력
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # lr 업데이트
        lr_scheduler.step()
        # 테스트 데이터로 평가
        evaluate(model, data_loader_test, device=device)

    print("That's it!")


if __name__ == '__main__':
    main()
