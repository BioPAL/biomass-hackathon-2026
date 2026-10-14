from typing import Callable
import pyvista as pv


class BasicPlotterApp:
    MAIN = "terrain_main"
    CONTOURS = "terrain_contours"

    def __init__(self):
        self._first_render = True

        self.pl = pv.Plotter()
        self.button_y = 500
        self.buttons = {}
        self.slider = None

        self.build_ui()
        self.render()

    def build_ui(self):
        raise NotImplementedError("Subclasses must implement _build_ui()")

    def increase_btn_y(self):
        self.button_y -= 45
        return self.button_y

    def add_checkbox(
        self, btn_id: str, label: str, callback: Callable, value: bool = False
    ):
        y = self.increase_btn_y()
        self.buttons[btn_id] = {
            "value": value,
            "button": self.pl.add_checkbox_button_widget(
                callback,
                value=value,
                position=(15, y),
                size=30,
                border_size=3,
                color_on="#2b7a4b",
                color_off="#d9d9d9",
                background_color="#555555",
            ),
        }
        self.pl.add_text(
            label,
            position=(55, y + 5),
            font_size=11,
            color="black",
            name=f"label_{label}",
        )
        self.pl.add_key_event(str(y + 1), lambda m=btn_id: callback(m))

    def sync_buttons(self):
        for button_info in self.buttons.values():
            button = button_info["button"]
            value = button_info["value"]
            button.GetRepresentation().SetState(int(value))

    def on_render(self):
        raise NotImplementedError("Subclasses must implement on_render()")

    def render(self):
        pl = self.pl
        pl.clear_plane_widgets()
        for name in (self.MAIN, self.CONTOURS):
            pl.remove_actor(name, render=False)
        self.sync_buttons()

        self.on_render()

        pl.add_axes()
        pl.show_grid()
        pl.render()

    def run(self):
        self.pl.show()
