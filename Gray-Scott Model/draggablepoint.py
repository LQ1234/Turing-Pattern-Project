import time

# DraggablePoint class definition with default value and callback throttling
class DraggablePoint:
    def __init__(self, ax, xlim=(0, 1), ylim=(0, 1), update_callback=None, default_value=(0.5, 0.5), callback_interval=0.1):
        self.ax = ax
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.update_callback = update_callback
        self.is_pressed = False
        self.callback_interval = callback_interval
        self.last_callback_time = time.time()  # To track when the last callback was made
        
        self.ax.set_aspect('equal', adjustable='box')
        
        # Initialize point at the default position
        default_x = default_value[0]
        default_y = default_value[1]
        self.point, = ax.plot([default_x], [default_y], 'ro', markersize=10)
        
        self.cidpress = self.point.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.point.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.point.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        if event.inaxes != self.point.axes:
            return
        contains, _ = self.point.contains(event)
        if contains:
            self.is_pressed = True

    def on_release(self, event):
        self.is_pressed = False
        self.point.figure.canvas.draw()

    def on_motion(self, event):
        if not self.is_pressed or event.inaxes != self.point.axes:
            return

        new_x = event.xdata
        new_y = event.ydata

        self.point.set_data(new_x, new_y)
        self.point.figure.canvas.draw()

        # Throttle the callback to only call every `callback_interval` seconds
        current_time = time.time()
        if current_time - self.last_callback_time > self.callback_interval:
            self.last_callback_time = current_time
            if self.update_callback:
                self.update_callback(new_x, new_y)