import torch
import argparse
import os.path
import numpy as np
import cv2
import binascii

def rgb_to_iyu2(acts):
    b, c, h, w = acts.shape
    iyu2 = torch.zeros_like(acts)
    assert c == 3

    for bn, act in enumerate(acts):
        r, g, b = act[0], act[1], act[2]
        y = 0.213 * r + 0.715 * g + 0.072 * b
        u = -0.117 * r - 0.394 * g + 0.511 * b + 128
        v = 0.511 * r - 0.464 * g - 0.047 * b + 128
        iyu2[bn, 0], iyu2[bn, 1], iyu2[bn, 2] = u, y, v

    return iyu2.round().clamp(0, 255)


def print_act_mem(acts, fn, little_end=True, word_align=True, align=1):

    b, c, y, x = acts.shape

    if word_align:
        if c % 32 is not 0:
            dummy = torch.zeros([1, 1, y, x], device=acts.device)
            for i in range(32 - (c % 32)):
                acts = torch.cat((acts, dummy), 1)

    assert align > 0
    stride = int((x + align - 1) / align)*align

    acts = acts.permute(0, 2, 3, 1)         # b,y,x,c
    if word_align:
        acts = acts.view(b, y, x, -1, 32)   # b,y,x,c/N,N
        acts = acts.permute(0, 3, 1, 2, 4)  # b,c/N,y,x,N
    else:
        acts = acts.unsqueeze(dim=1)

    acts = acts.data.cpu().numpy()

    with open(fn, "w") as f:
        for b_ in acts:
            for c_ in b_:
                for y_ in c_:
                    for x_ in y_:     # w
                        str_n = ''
                        for d in x_:  # c
                            s = hex(d.astype(int) & (2**8-1)).lstrip('0x').zfill(2)
                            if little_end:
                                str_n = '{}{}'.format(s, str_n)
                            else:
                                str_n = '{}{}'.format(str_n, s)
                        print(str_n, file=f)

                    str_n = '00'*len(y_[0])
                    for _ in range(len(y_), stride):
                        print(str_n, file=f)


def generate_bin(read_fn, endian='little'):
    r''' convert text to binary file'''

    filename, ext = os.path.splitext(read_fn)
    write_file_name = filename + '.bin'

    print('{} created'.format(write_file_name))

    with open(read_fn) as rf, open(write_file_name, 'wb') as wf:
        lines = rf.readlines()
        for line in lines:
            if endian == 'little':
                binary_data = "".join(reversed([line[i:i + 2]
                                      for i in range(0, len(line), 2)]))
                binary_data = binary_data.strip('\n')
            else:
                binary_data = line.strip('\n')
            wf.write(binascii.unhexlify(binary_data))


def parse_args():
    
    parser = argparse.ArgumentParser(description='convert img to yuv')

    parser.add_argument('-i', '--input', type=str,
                        default='./sample/zidane.jpg', metavar='FILENAME',
                        help='input image filename, default ./sample/zidane.jpg')
    parser.add_argument('-o', '--output', type=str,
                        default='./output/Relu_0.ia.bin', metavar='FILENAME',
                        help='output image filename, default ./output/Relu_0.ia.bin')
    parser.add_argument('--img-size', type=int,
                        default=300, metavar='TARGET_SIZE', help='resize img size')
    parser.add_argument('--rgb', action='store_true',
                        help='Enable resized rgb out')
    args = parser.parse_args()
    return args


def gen_ia_bin(args):
    print(f'input name: {args.input}')
    print(f'out name: {args.output}')
    assert os.path.exists(args.input), f'{args.input} not found'

    # BGR -> RGB
    input = cv2.imread(args.input)[:, :, ::-1]

    output = cv2.imread(args.input)
    input_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    # src image size to target image size
    input = cv2.resize(input, (args.img_size, args.img_size))

    # To Tensor [1, C, H, W]
    input = torch.from_numpy(input.astype(np.float32)).permute(2, 0, 1)
    input = input.unsqueeze_(dim=0)
    if torch.cuda.is_available():
        input = input.cuda()

    out_fn = os.path.splitext(args.output)[0]
    out_dir = os.path.dirname(out_fn)
    os.path.exists(out_dir) or os.mkdir(out_dir)

    #RGB dump
    if args.rgb:
        fn = f'{out_fn}.rgb.txt'
        print_act_mem(input, fn, word_align=False, align=16)
        generate_bin(fn)

    # UYV2 dump
    input = rgb_to_iyu2(input)
    fn = f'{out_fn}.txt'
    print_act_mem(input, fn, word_align=False, align=16)
    generate_bin(fn)

    assert os.path.exists(fn), f'{fn} not found'
    bin_fn = os.path.splitext(fn)[0] + '.bin'
    assert os.path.exists(bin_fn), f'{bin_fn} not found'
    print('Done')
                        

if __name__ == '__main__':
    args = parse_args()
    gen_ia_bin(args)
