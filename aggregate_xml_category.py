import os
from collections import Counter
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt


_ANNOTETION_DIR = '/media/panasonic/644E9C944E9C611A/tmp/data/detection/20190213_gyoza/annotation'
_TARGET_DIR = '/media/panasonic/644E9C944E9C611A/tmp/data/detection/20190213_gyoza/'


def main():
    categories = []
    xml_files = os.listdir(_ANNOTETION_DIR)
    for xml_file in xml_files:
        xml_file = os.path.join(_ANNOTETION_DIR, xml_file)
        tree = ET.ElementTree(file=xml_file)
        root = tree.getroot()

        for items in root.findall('.object'):
            for idx, item in enumerate(items):
                if str(item.tag) == 'name':
                    # print(item.tag)
                    # print(item.text)
                    categories.append(item.text)

    c = Counter(categories)
    print(c)
    names = []
    counts = []
    for key, value in c.items():
        names.append(key)
        counts.append(value)

    plt.figure()
    plt.bar(names, counts)
    save_filepath = os.path.join(_TARGET_DIR, 'annotation_count.png')
    plt.savefig(save_filepath)
    plt.show()


if __name__ == '__main__':
    main()
