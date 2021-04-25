import torch
from decode_jpeg import decode_jpeg_cuda, read_file

import time
from PIL import Image
import numpy as np
import pickle
import paddle
import cv2

# img_path = '/workspace/datasets/horse2zebra/testA/n02381460_800.jpg'
img_path = '/workspace/datasets/data/ILSVRC2012/val/n10148035/ILSVRC2012_val_00000509.JPEG'

### time compare
repeat_times = 1005
warmup = 5

static_decode_time = 0

decode_time = 0
paddle_read_file_time = 0
paddle_decode_time = 0

torch_decode_time = 0
torch_read_file_time = 0
torch_odecode_time = 0

pil_time = 0
cv2_time = 0
start_gpu_decode_time = time.time()

for i in range(repeat_times):
    if i < warmup:
        cv2_time = 0
    start_time = time.time()
    e = cv2.imread(img_path)[...,::-1].astype('float32')

    f = paddle.to_tensor(e.transpose([2, 0, 1]))

    cv2_time += time.time() - start_time

for i in range(repeat_times):
    if i < warmup:
        decode_time = 0
        paddle_read_file_time = 0
        paddle_decode_time = 0

    start_time = time.time()
    a = paddle.io.read_file(img_path)
    mid_time = time.time()
    paddle_read_file_time += mid_time - start_time
    b = paddle.io.decode_jpeg(a)
    paddle_decode_time += time.time() - mid_time

    decode_time += time.time() - start_time

for i in range(repeat_times):
    if i < warmup:
        torch_decode_time = 0
        torch_read_file_time = 0
        torch_odecode_time = 0

    start_time = time.time()
    g = read_file(img_path)

    mid_time = time.time()
    torch_read_file_time += mid_time - start_time

    h = decode_jpeg_cuda(g, 0)
    
    torch_odecode_time += time.time() - mid_time
    torch_decode_time += time.time() - start_time

for i in range(repeat_times):
    if i < warmup:
        pil_time = 0
    start_time = time.time()
    c = np.array(Image.open(img_path).convert('RGB')).astype('float32')
    d = paddle.to_tensor(c.transpose([2, 0, 1]))

    pil_time += time.time() - start_time

# paddle.enable_static()
# a = paddle.io.read_file(img_path)
# b = paddle.io.decode_jpeg(a)

# place = paddle.CUDAPlace(0)
# exe = paddle.static.Executor(place)
# # startup_program = paddle.static.Program()
# # main_program = paddle.static.Program()
# exe.run(paddle.static.default_startup_program())

# for i in range(repeat_times):
#     if i < warmup:
#         static_decode_time = 0

#     start_time = time.time()

#     out = exe.run(paddle.static.default_main_program(), fetch_list=[b.name])

#     static_decode_time += time.time() - start_time

# paddle.disable_static()

print('decode time: {:.5f} {:.5f} {:.5f}'.format(decode_time, paddle_read_file_time, paddle_decode_time), 'pil time: {:.5f}'.format(pil_time), 'cv2 time: {:.5f}'.format\
        (cv2_time), 'torch decode time: {:.5f} {:.5f} {:.5f}'.format(torch_decode_time, torch_read_file_time, torch_odecode_time), 'static name: {:.5f}'.format(static_decode_time))

print(out[0].shape)
# c = np.array(Image.open(img_path).convert('RGB'))
# d = paddle.to_tensor(c.transpose([2, 0, 1]))


# a = paddle.io.read_write_file.read_file('/workspace/datasets/ffhq/images1024x1024/69997.png')
# print(a)
# a = a.cuda()
# print('cuda a', a)
# print(b.numpy().transpose([1,2,0]))
# pickle.dump(b.numpy().transpose([1,2,0]), open('img_decode.pkl', 'wb'))
# print(c)
# np.testing.assert_allclose(b.numpy().transpose([1,2,0]), c)

# decode time: 0.3908073902130127 pil time: 0.314239501953125

