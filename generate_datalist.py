import os

image_dir = '/media/tsukudamayo/0CE698DCE698C6FC/tmp/data/dataset/gyoza_20190203_00-04/annotation/annotation/image/'
example_dir = '/media/tsukudamayo/0CE698DCE698C6FC/tmp/data/dataset/gyoza_20190203_00-04/annotation/annotation/example'

if os.path.exists(example_dir) is False:
    os.mkdir(example_dir)

print(os.listdir(image_dir))

example_txt = os.path.join(example_dir, 'cooking_example.txt')

with open(example_txt, 'w') as w:
    for idx, f in enumerate(os.listdir(image_dir)):
        name, ext = os.path.splitext(f)
        print(name)
        w.write(name + ' ' + str(idx))
        w.write('\n')
