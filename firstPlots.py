import os
import pandas as pd
import plotly.graph_objects as go
import tkinter as tk
from tkinter import ttk
import webbrowser


class ChannelVisualizer:
    def __init__(self, filepath):
        """
        Initialize the visualizer with data file path

        Parameters:
        -----------
        filepath : str
            Path to the CSV file containing channel data
        """
        # Load data
        self.df = pd.read_csv(filepath)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.channels = [col for col in self.df.columns if col.startswith('channel_')]
        # Store the file name for display in the plot
        self.filename = os.path.basename(filepath)

        # Create GUI
        self.root = tk.Tk()
        self.root.title("Channel Visualization")
        self.create_gui()

    def create_gui(self):
        """Create the GUI interface"""
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        select_frame = ttk.LabelFrame(main_frame, text="Select Channels", padding="5")
        select_frame.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))

        self.channel_listbox = tk.Listbox(
            select_frame,
            selectmode=tk.MULTIPLE,
            height=10,
            width=30
        )
        self.channel_listbox.grid(row=0, column=0, padx=5, pady=5)

        scrollbar = ttk.Scrollbar(select_frame, orient=tk.VERTICAL)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.channel_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.channel_listbox.yview)

        for channel in self.channels:
            self.channel_listbox.insert(tk.END, channel)

        # Select first channel by default
        self.channel_listbox.selection_set(0)

        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=1, column=0, padx=5, pady=5)

        ttk.Button(
            button_frame,
            text="Update Plot",
            command=self.update_plot
        ).grid(row=0, column=0, padx=5)

        ttk.Button(
            button_frame,
            text="Select All",
            command=self.select_all_channels
        ).grid(row=0, column=1, padx=5)

        ttk.Button(
            button_frame,
            text="Clear Selection",
            command=self.clear_selection
        ).grid(row=0, column=2, padx=5)

    def select_all_channels(self):
        """Select all channels in the listbox"""
        self.channel_listbox.select_set(0, tk.END)

    def clear_selection(self):
        """Clear all channel selections"""
        self.channel_listbox.selection_clear(0, tk.END)

    def create_plot(self, selected_channels):
        """Create the Plotly figure with selected channels"""
        fig = go.Figure()

        # Add traces for each selected channel
        for channel in selected_channels:
            fig.add_trace(
                go.Scatter(
                    x=self.df['timestamp'],
                    y=self.df[channel],
                    name=channel,
                    mode='lines',
                    hovertemplate=f'{channel}: %{{y:.2f}}<extra></extra>'
                )
            )

        # Set unified hover to capture all channels at a timestamp
        fig.update_layout(
            title=f'Channel Data Visualization - {self.filename}',
            xaxis_title='Time',
            yaxis_title='Value',
            hovermode='x unified',
            height=800,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(255, 255, 255, 0.8)'
            ),
            margin=dict(t=100),
            updatemenus=[dict(
                type="buttons",
                direction="right",
                x=0.7,
                y=1.2,
                buttons=[
                    dict(
                        args=[{"visible": [True] * len(selected_channels)}],
                        label="Show All",
                        method="restyle"
                    ),
                    dict(
                        args=[{"visible": [False] * len(selected_channels)}],
                        label="Hide All",
                        method="restyle"
                    ),
                ]
            )],
            hoverlabel=dict(
                bgcolor="rgba(255,255,255,0.8)",
                font_size=12,
                font_family="Arial"
            )
        )

        # Add range slider
        fig.update_xaxes(rangeslider_visible=True)

        fig.update_yaxes(fixedrange=False)

        # Configure spike lines for improved reading
        fig.update_layout(
            xaxis=dict(
                showspikes=True,
                spikemode='across',
                spikesnap='cursor',
                showline=True,
                showgrid=True,
                spikecolor="grey",
                spikethickness=1
            ),
            yaxis=dict(
                showspikes=True,
                spikemode='across',
                spikesnap='cursor',
                showline=True,
                showgrid=True,
                spikecolor="grey",
                spikethickness=1
            )
        )

        return fig

    def update_plot(self):
        """Update and display the plot with selected channels"""
        selected_indices = self.channel_listbox.curselection()
        selected_channels = [self.channels[i] for i in selected_indices]

        if not selected_channels:
            selected_channels = [self.channels[0]]
            self.channel_listbox.selection_set(0)

        fig = self.create_plot(selected_channels)

        # The post_script does the following:
        # 1. Hides Plotly's default hover labels.
        # 2. Creates a fixed div for displaying hover info.
        # 3. Captures the timestamp from the unified hover event.
        # 4. Listens for the spacebar press; when pressed, it formats the time
        #    (HH:MM:SS.microseconds) from the hovered timestamp and copies it to the clipboard.
        post_script = """
// Global variable to hold the current hovered timestamp
var currentHoverTimestamp = null;

// Function to format timestamp to HH:MM:SS.microseconds.
// Note: JavaScript Date only provides milliseconds, so we simulate microseconds.
function formatTime(timestamp) {
    var date = new Date(timestamp);
    var hours = String(date.getHours()).padStart(2, '0');
    var minutes = String(date.getMinutes()).padStart(2, '0');
    var seconds = String(date.getSeconds()).padStart(2, '0');
    var milliseconds = String(date.getMilliseconds()).padStart(3, '0');
    // Append "000" to represent microseconds (since we only have milliseconds)
    return hours + ":" + minutes + ":" + seconds + "." + milliseconds + "000";
}

// Hide default hover tooltips via CSS
var style = document.createElement('style');
style.innerHTML = ".hoverlayer { display: none !important; }";
document.head.appendChild(style);

// Create a fixed div to show hover info if not already present
if (!document.getElementById('fixed-hover')) {
    var fixedHover = document.createElement('div');
    fixedHover.id = 'fixed-hover';
    fixedHover.style.position = 'fixed';
    fixedHover.style.top = '10px';
    fixedHover.style.right = '10px';
    fixedHover.style.background = 'white';
    fixedHover.style.border = '1px solid black';
    fixedHover.style.padding = '10px';
    fixedHover.style.zIndex = '9999';
    document.body.appendChild(fixedHover);
}

var plotDiv = document.getElementById('plotly-graph');

// Listen for hover events: update fixed div and save hovered timestamp.
plotDiv.on('plotly_hover', function(data){
    currentHoverTimestamp = data.points[0].x;
    var infotext = ["Timestamp: " + data.points[0].x];
    infotext = infotext.concat(data.points.map(function(d){
        return d.data.name + ': ' + d.y.toFixed(2);
    }));
    document.getElementById('fixed-hover').innerHTML = infotext.join('<br>');
});

// Clear fixed div and timestamp on unhover.
plotDiv.on('plotly_unhover', function(data){
    currentHoverTimestamp = null;
    document.getElementById('fixed-hover').innerHTML = '';
});

// Listen for spacebar keydown to copy the time portion to the clipboard.
window.addEventListener('keydown', function(event) {
    if (event.code === 'Space' || event.key === ' ') {
        if (currentHoverTimestamp !== null) {
            var timeStr = formatTime(currentHoverTimestamp);
            navigator.clipboard.writeText(timeStr+',').then(function() {
                console.log('Time copied to clipboard:', timeStr);
            }).catch(function(err) {
                console.error('Error copying time to clipboard:', err);
            });
        }
    }
});
"""

        html_path = 'channel_visualization.html'
        fig.write_html(
            html_path,
            include_plotlyjs='cdn',
            post_script=post_script,
            div_id='plotly-graph'
        )
        webbrowser.open('file://' + os.path.realpath(html_path))

    def run(self):
        """Start the visualization application"""
        self.root.mainloop()


def visualize_channels(filepath):
    """
    Create and display the channel visualization

    Parameters:
    -----------
    filepath : str
        Path to the CSV file containing channel data
    """
    app = ChannelVisualizer(filepath)
    app.run()


if __name__ == "__main__":
    # Replace with your file path
    filepath = os.path.join("23-2", "blinks.csv")
    visualize_channels(filepath)
