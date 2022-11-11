from torchvision import transforms
from torch.autograd import Variable
from torchvision import utils
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def get_torch_variable(arg, cuda=True, cuda_index=0):
    if cuda:
        return Variable(arg).cuda(cuda_index)
    else:
        return Variable(arg)

transform = transforms.Compose([
transforms.Resize([64, 64]),
# transforms.Grayscale(3),
# CustomColorChange(colors=colors, all_random=False, debug=debug),
transforms.ToTensor(),
transforms.Normalize((0.5, ), (0.5, ))
])

def visualize_feature_map(conv_layers):
    # z = Variable(Tensor(np.random.normal(0, 1, (1, 100)))).cuda()
    z = Image.open('input.jpg')
    z = transform(z).cuda()
    z = z.unsqueeze(1).cuda()
    outputs = []
    names = []
    i = 0
    for layer in conv_layers[0:]:
        print(str(layer), end='')
        z = layer(z)
        image = z.mul(0.5).add(0.5)
        image = image.data[:64]
        outputs.append(image)
        names.append(str(layer))
        print(image)
        grid = utils.make_grid(image)
        utils.save_image(grid, 'TEST-layer{}.png'.format(i))
        i+=1
    # samples = generator(z)
    # samples = samples.mul(0.5).add(0.5)
    # samples = samples.data.cpu()[:64]
    # grid = utils.make_grid(samples)
    # utils.save_image(grid, '{}/training_result_images/img_generatori_iter_{}_pid_{}.png'.format(root, str(g_iter).zfill(3), id))

    # latent_dim = 100
    # batch_size = 64
    # image = Variable(Tensor(np.random.normal(0, 1, (batch_size, latent_dim)))).cuda()
    # outputs = []
    # names = []
    # i = 0
    # for layer in conv_layers[0:]:
    #     print(i)
    #     image = layer(image)
    #     outputs.append(image)
    #     names.append(str(layer))
    # # print(len(outputs))
    # # #print feature_maps
    # # for feature_map in outputs:
    # #     print(feature_map.shape)

    # processed = []
    # for feature_map in outputs:
    #     # feature_map = feature_map.squeeze(0)
    #     gray_scale = torch.sum(feature_map,0)
    #     gray_scale = gray_scale / feature_map.shape[0]
    #     processed.append(gray_scale.data.cpu().numpy())
    # # for fm in processed:
    # #     print(fm.shape)

    # fig = plt.figure(figsize=(30, 50))
    # for i in range(len(processed)):
    #     a = fig.add_subplot(5, 4, i+1)
    #     # imgplot = plt.imshow(processed[i])
    #     a.axis("off")
    #     a.set_title(names[i].split('(')[0], fontsize=30)
    # plt.savefig(str('feature_maps.jpg'), bbox_inches='tight')