import matplotlib.pyplot as plt


def plot_results(x_axis_vals, size, y_axis_vals):
    fig, ax = plt.subplots()
    ax.plot(x_axis_vals, y_axis_vals, label='Accuracy')
    ax.set(xlabel='Depth', ylabel='Accuracy %',
           title='Accuracy vs Depth for Ensemble size {}'.format(size))
    ax.legend()
    for xy in zip(x_axis_vals, y_axis_vals):
        ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')
    fig.savefig('../data/problem_1.2_accuracy_{}.png'.format(size))


if __name__ == '__main__':
    res = {
        1: [
            60.8,
            72.39,
            71.6,
            73.2,
        ],
        3: [
            72.0,
            74.0,
            73.6,
            74.8
        ],
        5: [
            70.8,
            73.6,
            73.6,
            73.8
        ],
        10: [
            70.8,
            73.2,
            72.0,
            73.39
        ],
        20: [
            72.0,
            72.0,
            72.2,
            74.4
        ]
    }
    for k, v in res.items():
        plot_results(x_axis_vals=[1, 2, 3, 6], y_axis_vals=v, size=k)
