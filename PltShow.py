import matplotlib.pyplot as plt
import random


def show_result(result):
    fig = plt.figure('识别结果')

    for num, data in enumerate(result):
        probability = data[0]
        print(probability)
        im = data[1]
        label = data[2]
        y = fig.add_subplot(4, 4, num + 1)
        y.imshow(im, cmap='gray')
        # ran = random.uniform(0, 0.4)
        # t = float(probability)
        plt.title(label + ' '+ str(probability)[:6])
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)
    plt.show()