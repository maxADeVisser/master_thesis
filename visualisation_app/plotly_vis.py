# Import data
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from skimage import io


def plotly_fig():
    """Create a 3D Plotly figure with a slider to navigate through MRI slices"""
    vol = io.imread(
        "https://s3.amazonaws.com/assets.datacamp.com/blog_assets/attention-mri.tif"
    )
    volume = vol.T
    nb_frames, r, c = volume.shape

    fig = go.Figure(
        frames=[
            go.Frame(
                data=go.Surface(
                    z=(6.7 - k * 0.1) * np.ones((r, c)),
                    surfacecolor=np.flipud(volume[67 - k]),
                    cmin=0,
                    cmax=200,
                ),
                name=str(
                    k
                ),  # you need to name the frame for the animation to behave properly
            )
            for k in range(nb_frames)
        ]
    )

    # Add data to be displayed before animation starts
    fig.add_trace(
        go.Surface(
            z=6.7 * np.ones((r, c)),
            surfacecolor=np.flipud(volume[67]),
            colorscale="Gray",
            cmin=0,
            cmax=200,
            colorbar=dict(thickness=20, ticklen=4),
        )
    )

    def frame_args(duration):
        return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

    sliders = [
        {
            "pad": {"b": 10, "t": 60},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [[f.name], frame_args(0)],
                    "label": str(k),
                    "method": "animate",
                }
                for k, f in enumerate(fig.frames)
            ],
        }
    ]

    # Layout
    fig.update_layout(
        title="Slices in volumetric data",
        width=800,
        height=800,
        scene=dict(
            zaxis=dict(range=[-0.1, 6.8], autorange=False),
            aspectratio=dict(x=1, y=1, z=1),
        ),
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, frame_args(50)],
                        "label": "&#9654;",  # play symbol
                        "method": "animate",
                    },
                    {
                        "args": [[None], frame_args(0)],
                        "label": "&#9724;",  # pause symbol
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "type": "buttons",
                "x": 0.1,
                "y": 0,
            }
        ],
        sliders=sliders,
    )
    return fig


def plotly_fig2(volume: np.ndarray) -> go.Figure:
    # Normalize the data if needed
    # This step might be necessary if your image data isn't already normalized to [0, 255]
    # volume = 255 * (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
    # volume = volume.astype(np.uint8)  # Convert to 8-bit unsigned integer

    nb_frames, r, c = volume.shape
    fig = go.Figure(
        frames=[
            go.Frame(
                data=go.Heatmap(
                    z=np.flipud(volume[k]),  # Flipped slice for visualization
                    zmin=0,
                    zmax=255,
                    colorscale="gray",  # You can choose a different colorscale
                ),
                name=str(k),  # Name the frame for proper animation behavior
            )
            for k in range(nb_frames)
        ]
    )

    # Add the first frame as the initial data for the figure
    fig.add_trace(
        go.Heatmap(
            z=np.flipud(volume[0]),
            zmin=0,
            zmax=255,
            colorscale="gray",
        )
    )

    # Set up layout, including a slider for controlling the animation
    fig.update_layout(
        title="2D CT Scan Slices",
        xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Slice: "},
                "pad": {"b": 10, "t": 100},
                "steps": [
                    {
                        "label": str(k),
                        "method": "animate",
                        "args": [[str(k)], {"mode": "immediate"}],
                    }
                    for k in range(nb_frames)
                ],
            }
        ],
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [
                            None,
                            {
                                "frame": {"duration": 100, "redraw": True},
                                "fromcurrent": True,
                            },
                        ],
                        "label": "Play",
                        "method": "animate",
                    },
                    {
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": "Pause",
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top",
            }
        ],
        width=800,
        height=800,
    )

    return fig
