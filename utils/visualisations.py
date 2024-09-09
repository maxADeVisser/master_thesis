import matplotlib.pyplot as plt
from matplotlib import animation, rc

rc("animation", html="jshtml")


# see https://www.kaggle.com/code/grolakr/hesaplanabilir-sinirbilim-proje-2
def create_animation(ims):
    fig = plt.figure(figsize=(6, 6))
    plt.axis("off")
    im = plt.imshow(ims[0], cmap="gray")

    def animate_func(i):
        im.set_array(ims[i])
        return [im]

    return animation.FuncAnimation(
        fig, animate_func, frames=len(ims), interval=1000 // 24
    )
