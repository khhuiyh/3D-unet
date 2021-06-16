import nibabel as nib
import os
from Unet_model import *
from process_fun import *
from skimage.measure import label, regionprops
import csv
import random

maxnum = 19000


def write_nii(input_filename, input_data, output_filename):
    # 读取nii文件
    nii_img = nib.load(input_filename)
    new_data = input_data

    affine = nii_img.affine.copy()
    hdr = nii_img.header.copy()

    # 形成新的nii文件
    new_nii = nib.Nifti1Image(new_data, affine, hdr)

    nib.save(new_nii, output_filename)


def printpath(level, path):
    filelistt = []
    files = os.listdir(path)
    for f in files:
        if os.path.isfile(path + '/' + f):
            filelistt.append(f)
    return filelistt


def get_outdata(input_data, minarea, pro_data, img_data, pixel_threshold=-100.0):
    output = label(input_data).astype(int)
    output_copy = output.copy()
    props = regionprops(output)

    c = 1
    count = 0
    confidence_list = [np.average((1 - pro_data)[output == 0])]
    for i in props:
        flag = (output == c)
        if i.area <= minarea or np.average(img_data[flag]) <= pixel_threshold:
            output_copy[flag] = 0
        else:
            count += 1
            print(count,'area:', i.area)
            print('pixel average:', np.average(img_data[flag]))
            output_copy[flag] = count
            print('pred average:', np.average(pro_data[flag]))
            confidence_list.append(np.average(pro_data[flag]))
        c += 1

    return output_copy, confidence_list


def dataprocess(orgindata, output, pixel_threshold=-100.0):
    myout = output
    dim0, dim1, dim2 = 16, 16, 16
    for start2 in range(0, x0.size(0), dim2):
        stop2 = min(start2 + dim2 - 1, x0.size(0) - 1)
        for start1 in range(0, x0.size(2), dim1):
            stop1 = min(start1 + dim1 - 1, x0.size(2) - 1)
            for start0 in range(0, x0.size(1), dim0):
                stop0 = min(start0 + dim0 - 1, x0.size(1) - 1)
                batchx = orgindata[start2:stop2 + 1, start0:stop0 + 1, start1:stop1 + 1]
                if np.average(batchx) <= pixel_threshold:
                    myout[start2:stop2 + 1, start0:stop0 + 1, start1:stop1 + 1] = 0.0
    return myout


if __name__ == '__main__':
    a = 'w'  # a继续，w从头来
    path_checkpoint = r"E:\dataset\checkpoint\checkpoints_loss_0.5360.pth"  # 断点路径
    areathreshold = 500
    pred_shold = 0.2
    pixelthreshold = -100
    if a == 'w':
        num_image = 0
    else:
        num_image = np.load('pre_savepoint.npz')['arr_0'] + 1
    csv_filename = './prediction_test/ribfrac-test-pred.csv'
    with open(csv_filename, a, newline='', encoding='utf-8') as f:
        csv_file = csv.writer(f)
        if a == 'w':
            csv_file.writerows([
                ['public_id', 'label_id', 'confidence', 'label_code'],
            ])
        dev = 'gpu'
        checkpoint = torch.load(path_checkpoint)  # 加载断点
        model = thrdunet(in_channels=1, out_channels=1, num_conv_blocks=2, model_depth=3, dev=dev)
        # model = mythrdunet_transpose()
        model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
        if dev == 'gpu':
            model = model.cuda()
        filelist_val = printpath(1, "./test/ribfrac-test-images/")[:]  # test集data路径
        for i in range(num_image, len(filelist_val)):
            file = filelist_val[i]
            data_filename = './test/ribfrac-test-images/' + file
            outdir = './prediction_test/' + file.split('-')[0] + '.nii.gz'

            num_image = num_image + 1
            print('num_image', num_image)
            origin_data = nib.load(data_filename).get_fdata()[:]
            inputshape = origin_data.shape
            img = minmax_normalize(origin_data)
            x0 = torch.tensor(img)
            out = torch.zeros_like(x0)

            dim0, dim1, dim2 = 64, 64, 64

            step = 0
            for start2 in range(0, x0.size(0), 32):
                stop2 = min(start2 + dim2 - 1, x0.size(0) - 1)
                if stop2 - start2 + 1 <= 63:
                    # out[start2:stop2 + 1, :, :] = 0
                    out[start2:stop2 + 1, :, :] += 0
                    continue
                for start1 in range(0, x0.size(2), 32):
                    stop1 = min(start1 + dim1 - 1, x0.size(2) - 1)
                    if stop1 - start1 + 1 <= 63:
                        # out[start2:stop2 + 1, :, start1:stop1 + 1] = 0
                        out[start2:stop2 + 1, :, start1:stop1 + 1] += 0
                        continue
                    for start0 in range(0, x0.size(1), 32):
                        stop0 = min(start0 + dim0 - 1, x0.size(1) - 1)
                        if stop0 - start0 + 1 <= 63:
                            # out[start2:stop2 + 1, start0:stop0 + 1, start1:stop1 + 1] = 0
                            out[start2:stop2 + 1, start0:stop0 + 1, start1:stop1 + 1] += 0
                            continue
                        batch_x = x0[start2:stop2 + 1, start0:stop0 + 1, start1:stop1 + 1].float()
                        batch_x = torch.unsqueeze(batch_x, dim=0)
                        batch_x = torch.unsqueeze(batch_x, dim=0)
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                        if torch.cuda.is_available() and dev == 'gpu':
                            batch_x = batch_x.cuda()
                        with torch.no_grad():
                            y_pred = model.forward(batch_x).cpu()
                        # print(torch.max(y_pred), torch.min(y_pred))
                        step += 1
                        # print('num_image', num_image, '| step', step)
                        # out[start2:stop2 + 1, start0:stop0 + 1, start1:stop1 + 1] = torch.squeeze(y_pred)
                        out[start2:stop2 + 1, start0:stop0 + 1, start1:stop1 + 1] += torch.squeeze(y_pred)
            out = out / 8.0
            data_pred = dataprocess(origin_data, np.array(out), pixelthreshold)
            databinary = data_pred.copy()
            pp = data_pred.copy()
            flagflag = (pp >= pred_shold)
            databinary[flagflag] = 1
            flagflag = (pp < pred_shold)
            databinary[flagflag] = 0

            outputshape = data_pred.shape
            out_data, list_confidence = get_outdata(databinary, areathreshold, data_pred, origin_data, pixelthreshold)

            write_nii(data_filename, out_data, outdir)
            csv_file.writerows([
                [file.split('-')[0], 0, list_confidence[0], 0]
            ])
            for k in range(1, len(list_confidence)):
                csv_file.writerows([
                    [file.split('-')[0], k, list_confidence[k], 1]
                ])
            np.save('pre_savepoint.npz', i)
            print('figure: ', num_image, ' finish', '  input shape:', inputshape, 'outputshape', outputshape)

    # a = 'w'  # a继续，w从头来
    # path_checkpoint = r"E:\dataset\checkpoint\checkpoints_loss_0.4364.pth"  # 断点路径
    # areathreshold = 1000
    # pred_shold = 0.2
    # pixelthreshold = -100
    # if a == 'w':
    #     num_image = 0
    # else:
    #     num_image = np.load('pre_savepoint.npz')['arr_0'] + 1
    # csv_filename = './prediction/ribfrac-val-pred.csv'
    # with open(csv_filename, a, newline='', encoding='utf-8') as f:
    #     csv_file = csv.writer(f)
    #     if a == 'w':
    #         csv_file.writerows([
    #             ['public_id', 'label_id', 'confidence', 'label_code'],
    #         ])
    #     dev = 'gpu'
    #     checkpoint = torch.load(path_checkpoint)  # 加载断点
    #     # model = thrdunet(in_channels=1, out_channels=1, num_conv_blocks=2, model_depth=3, dev=dev)
    #     model = mythrdunet_transpose()
    #     model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
    #     if dev == 'gpu':
    #         model.cuda()
    #     filelist_val = printpath(1, "E:/dataset/ribfrac-val/data")[:10]  # val集data路径
    #     for i in range(num_image, len(filelist_val)):
    #         file = filelist_val[i]
    #         label_filename = './ribfrac-val/label/' + file.split('-')[0] + '-label.nii.gz'
    #         data_filename = './ribfrac-val/data/' + file
    #         outdir = './prediction/' + file.split('-')[0] + '.nii.gz'
    #
    #         num_image = num_image + 1
    #         print('num_image', num_image)
    #         origin_data = nib.load(data_filename).get_fdata()[:]
    #         inputshape = origin_data.shape
    #         img = minmax_normalize(origin_data)
    #         x0 = torch.tensor(img)
    #         out = torch.zeros_like(x0)
    #
    #         dim0, dim1, dim2 = 64, 64, 64
    #
    #         step = 0
    #         for start2 in range(0, x0.size(0), 32):
    #             stop2 = min(start2 + dim2 - 1, x0.size(0) - 1)
    #             if stop2 - start2 + 1 <= 63:
    #                 # out[start2:stop2 + 1, :, :] = 0
    #                 out[start2:stop2 + 1, :, :] += 0
    #                 continue
    #             for start1 in range(0, x0.size(2), 32):
    #                 stop1 = min(start1 + dim1 - 1, x0.size(2) - 1)
    #                 if stop1 - start1 + 1 <= 63:
    #                     # out[start2:stop2 + 1, :, start1:stop1 + 1] = 0
    #                     out[start2:stop2 + 1, :, start1:stop1 + 1] += 0
    #                     continue
    #                 for start0 in range(0, x0.size(1), 32):
    #                     stop0 = min(start0 + dim0 - 1, x0.size(1) - 1)
    #                     if stop0 - start0 + 1 <= 63:
    #                         # out[start2:stop2 + 1, start0:stop0 + 1, start1:stop1 + 1] = 0
    #                         out[start2:stop2 + 1, start0:stop0 + 1, start1:stop1 + 1] += 0
    #                         continue
    #                     batch_x = x0[start2:stop2 + 1, start0:stop0 + 1, start1:stop1 + 1].float()
    #                     batch_x = torch.unsqueeze(batch_x, dim=0)
    #                     batch_x = torch.unsqueeze(batch_x, dim=0)
    #                     if hasattr(torch.cuda, 'empty_cache'):
    #                         torch.cuda.empty_cache()
    #                     if torch.cuda.is_available() and dev == 'gpu':
    #                         batch_x = batch_x.cuda()
    #                     with torch.no_grad():
    #                         y_pred = model.forward(batch_x).cpu()
    #                     # print(torch.max(y_pred), torch.min(y_pred))
    #                     step += 1
    #                     # print('num_image', num_image, '| step', step)
    #                     # out[start2:stop2 + 1, start0:stop0 + 1, start1:stop1 + 1] = torch.squeeze(y_pred)
    #                     out[start2:stop2 + 1, start0:stop0 + 1, start1:stop1 + 1] += torch.squeeze(y_pred)
    #         out = out / 8.0
    #         data_pred = dataprocess(origin_data, np.array(out), pixelthreshold)
    #         databinary = data_pred.copy()
    #         pp = data_pred.copy()
    #         flagflag = (pp >= pred_shold)
    #         databinary[flagflag] = 1
    #         flagflag = (pp < pred_shold)
    #         databinary[flagflag] = 0
    #
    #         outputshape = data_pred.shape
    #         out_data, list_confidence = get_outdata(databinary, areathreshold, data_pred, origin_data,
    #                                                 pixelthreshold)
    #
    #         write_nii(label_filename, out_data, outdir)
    #         csv_file.writerows([
    #             [file.split('-')[0], 0, list_confidence[0], 0]
    #         ])
    #         for k in range(1, len(list_confidence)):
    #             csv_file.writerows([
    #                 [file.split('-')[0], k, list_confidence[k], 1]
    #             ])
    #         np.save('pre_savepoint.npz', i)
    #         print('figure: ', num_image, ' finish', '  input shape:', inputshape, 'outputshape', outputshape)
