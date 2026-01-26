import colorsys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap


def one_point_palette():
    """
    Red
    """
    return ['#CB6040']


def two_point_palette():
    """
    Teal, Red
    """
    return ['#257180', '#CB6040']


def three_point_palette():
    """
    Teal, Beige, Red
    """
    return ['#257180', '#F2E5BF', '#CB6040']


def four_point_palette():
    """
    Teal, Beige, Orange, Red
    """

    return ['#257180', '#F2E5BF', '#FD8B51', '#CB6040']


def five_point_palette():
    """
    The original palette with a colour blended between Teal and Beige
    """
    return ['#257180', '#8CABA0', '#F2E5BF', '#FD8B51', '#CB6040']


def plot_gradient_radar(ax, angles, mean_vals, lower_vals, upper_vals,
                        line_color, fill_color, label, n_layers=20):
    """Plot radar with bidirectional fading gradient"""

    # Close the polygons
    mean_closed = mean_vals + [mean_vals[0]]
    lower_closed = lower_vals + [lower_vals[0]]
    upper_closed = upper_vals + [upper_vals[0]]

    angles_arr = np.array(angles)
    mean_arr = np.array(mean_closed)
    lower_arr = np.array(lower_closed)
    upper_arr = np.array(upper_closed)

    # Upper CI gradient: from mean outward to upper bound
    for i in range(n_layers):
        t_inner = i / n_layers
        t_outer = (i + 1) / n_layers

        inner_ring = mean_arr + (upper_arr - mean_arr) * t_inner
        outer_ring = mean_arr + (upper_arr - mean_arr) * t_outer

        alpha = 0.9 * (1 - t_outer / 1.2)

        ax.fill_between(angles_arr, inner_ring, outer_ring,
                        color=fill_color, alpha=alpha, linewidth=0, zorder=1)

    ax.plot(angles_arr, outer_ring, linewidth=1, color=line_color, label=None, zorder=2)

    # Lower CI gradient: from mean inward to lower bound
    for i in range(n_layers):
        t_inner = i / n_layers
        t_outer = (i + 1) / n_layers

        inner_ring = mean_arr + (lower_arr - mean_arr) * t_inner
        outer_ring = mean_arr + (lower_arr - mean_arr) * t_outer

        alpha = 1.00 * (1 - t_outer / 1.2)

        ax.fill_between(angles_arr, inner_ring, outer_ring,
                        color=fill_color, alpha=alpha, linewidth=0, zorder=1)

    ax.plot(angles_arr, outer_ring, linewidth=1, color=line_color, label=None, zorder=2)

    ax.plot(angles_arr, mean_arr, '-', linewidth=1.5, color='black',
            label=label, zorder=2)


def generate_color_variants(hex_color, num_variants=4, step: float = 0.25):
    """
    Generate brighter and darker variants of a given HEX color.

    Parameters
    ----------
    hex_color: str
        A base colour in HEX format to use
    num_variants: int
        Total number of NEW colours to generate
    step: float
        Step size to take on the colour wheel. Bigger step gives more different colours.

    Returns
    -------
    variants: List[str]
        Generated colours sorted from darkest to brightest

    Args:
        hex_color (str): The HEX colour code (with or without '#')
        num_variants (int): Total number of variants to generate (must be even)

    Returns:
        list: A list of color variants with original color in the middle
    """

    if num_variants % 2 != 0:
        num_variants += 1

    hex_color = hex_color.lstrip('#')

    # Convert hex to RGB
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0

    # Convert RGB to HSV
    h, s, v = colorsys.rgb_to_hsv(r, g, b)

    variants = []
    half_variants = num_variants // 2

    # Generate darker variants
    for i in range(half_variants, 0, -1):
        new_v = max(0.0, v * (1 - (i * step)))
        new_r, new_g, new_b = colorsys.hsv_to_rgb(h, s, new_v)

        hex_variant = "#{:02x}{:02x}{:02x}".format(
            int(new_r * 255),
            int(new_g * 255),
            int(new_b * 255)
        )
        variants.append(hex_variant)

    # Add original color
    variants.append(f"#{hex_color}")

    # Generate brighter variants
    for i in range(1, half_variants + 1):
        new_v = min(1.0, v * (1 + (i * step)))
        new_r, new_g, new_b = colorsys.hsv_to_rgb(h, s, new_v)

        hex_variant = "#{:02x}{:02x}{:02x}".format(
            int(new_r * 255),
            int(new_g * 255),
            int(new_b * 255)
        )
        variants.append(hex_variant)

    return variants
def set_font(font_filename: str = "arial.ttf"):
    """
    Register and set a custom TTF font globally for matplotlib.
    Works with both relative (from mpl data path) and absolute paths.
    """
    from pathlib import Path
    import matplotlib.pyplot as pl
    import matplotlib.font_manager as fm
    import matplotlib as mpl

    # Resolve font path
    font_path = (
        Path(mpl.get_data_path(), "fonts/ttf", font_filename)
        if not Path(font_filename).is_absolute()
        else Path(font_filename)
    )

    if not font_path.exists():
        print(f'Font file not found at: {font_path}. Using default font instead.')
    else:
        fm.fontManager.addfont(str(font_path))
        font_name = fm.FontProperties(fname=str(font_path)).get_name()
        pl.rcParams["font.family"] = font_name
        print(f"Matplotlib is now using: {font_name}")


def umap_plot(df: pd.DataFrame, fp_col: str, hue_col: str = None, umap_kwargs: dict = None, save_path: str = None, plot_kwargs: dict = None):
    """
    Prepare UMAP plot of passed dataframe.

    Parameters
    ----------
    df: pd.DataFrame
        Pandas Dataframe object.
    fp_col: str
        Name of the column with fingerprints.
    hue_col: str
        Name of the column to be used for coloring the plot.
    umap_kwargs: dict
        Dictionary holding parameters passed to the umap function.
    save_path: str
        If not None, path for saving the plot
    plot_kwargs: dict
        Additional parameters passed to seaborn relplot function.
    """
    random_state = np.random.RandomState(0)

    def_umap_kwargs = {
        'n_neighbors': 15,
        'min_dist': 0.1,
        'metric': 'jaccard'
    }

    if umap_kwargs is not None:
        def_umap_kwargs.update(**umap_kwargs)

    if plot_kwargs is None:
        plot_kwargs = {}

    df_ = df.copy()

    sns.set_context('notebook')
    sns.set_style('white')

    umap_ = umap.UMAP(random_state=random_state, **def_umap_kwargs)

    umap_emb = umap_.fit_transform(X=np.vstack(df_[fp_col].to_numpy()))

    df_['UMAP Component 0'] = umap_emb[:, 0]
    df_['UMAP Component 1'] = umap_emb[:, 1]

    if hue_col is not None:
        g = sns.relplot(df_, x='UMAP Component 0', y='UMAP Component 1', hue=hue_col, alpha=0.8, **plot_kwargs)
    else:
        g = sns.relplot(df_, x='UMAP Component 0', y='UMAP Component 1', alpha=0.8, **plot_kwargs)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')