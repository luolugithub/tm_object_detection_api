import os


image_dir = '/media/panasonic/644E9C944E9C611A/tmp/data/detection/20180911_gyoza_cookware/train/image'
example_dir = '/media/panasonic/644E9C944E9C611A/tmp/data/detection/20180911_gyoza_cookware/train/example'
if os.path.exists(example_dir) is False:
    os.mkdir(example_dir)

print(os.listdir(image_dir))

example_txt = os.path.join(example_dir, 'ingradient_example.txt')

with open(example_txt, 'w') as w:
    for idx, f in enumerate(os.listdir(image_dir)):
        name, ext = os.path.splitext(f)
        print(name)
        w.write(name + ' ' + str(idx))
        w.write('\n')
