from torchvision.transforms import Resize
from torchvision import transforms
import torch
import torch.nn.functional as F

from dicomogan.models.dino_extractor import VitExtractor


class SpliceLoss(torch.nn.Module):

    def __init__(self, structure_lambda=0.8):
        super().__init__()

        self.extractor = VitExtractor(model_name='dino_vitb16')
        for param in self.extractor.parameters():
            param.requires_grad = False

        self.structure_lambda = structure_lambda
        imagenet_norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        global_resize_transform = Resize(244)

        self.global_transform = transforms.Compose([global_resize_transform,
                                                    imagenet_norm
                                                    ])


    def forward(self, outputs, inputs):
        losses = {}
        loss_G = 0

        losses['loss_entire_ssim'] = self.calculate_global_ssim_loss(outputs, inputs)
        loss_G += self.structure_lambda * losses['loss_entire_ssim'] 


        losses['loss_entire_cls'] = self.calculate_crop_cls_loss(outputs, inputs)
        loss_G += (1 - self.structure_lambda) *  losses['loss_entire_cls'] 

        # if self.lambdas['lambda_global_cls'] > 0:
        #     losses['loss_global_cls'] = self.calculate_crop_cls_loss(outputs['x_global'], inputs['B_global'])
        #     loss_G += losses['loss_global_cls'] * self.lambdas['lambda_global_cls']

        # if self.lambdas['lambda_global_identity'] > 0:
        #     losses['loss_global_id_B'] = self.calculate_global_id_loss(outputs['y_global'], inputs['B_global'])
        #     loss_G += losses['loss_global_id_B'] * self.lambdas['lambda_global_identity']

        losses['loss'] = loss_G
        return loss_G

    def calculate_global_ssim_loss(self, outputs, inputs):
        loss = 0.0
        for a, b in zip(inputs, outputs):  # avoid memory limitations
            a = self.global_transform(a)
            b = self.global_transform(b)
            with torch.no_grad():
                target_keys_self_sim = self.extractor.get_keys_self_sim_from_input(a.unsqueeze(0), layer_num=11)
            keys_ssim = self.extractor.get_keys_self_sim_from_input(b.unsqueeze(0), layer_num=11)
            loss += F.mse_loss(keys_ssim, target_keys_self_sim)
        return loss

    def calculate_crop_cls_loss(self, outputs, inputs):
        loss = 0.0
        for a, b in zip(outputs, inputs):  # avoid memory limitations
            a = self.global_transform(a).unsqueeze(0).to(inputs.device)
            b = self.global_transform(b).unsqueeze(0).to(inputs.device)
            cls_token = self.extractor.get_feature_from_input(a)[-1][0, 0, :]
            with torch.no_grad():
                target_cls_token = self.extractor.get_feature_from_input(b)[-1][0, 0, :]
            loss += F.mse_loss(cls_token, target_cls_token)
        return loss

    def calculate_global_id_loss(self, outputs, inputs):
        loss = 0.0
        for a, b in zip(inputs, outputs):
            a = self.global_transform(a)
            b = self.global_transform(b)
            with torch.no_grad():
                keys_a = self.extractor.get_keys_from_input(a.unsqueeze(0), 11)
            keys_b = self.extractor.get_keys_from_input(b.unsqueeze(0), 11)
            loss += F.mse_loss(keys_a, keys_b)
        return loss