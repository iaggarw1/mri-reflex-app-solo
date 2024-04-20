import reflex as rx

from chat.components import loading_icon
from chat.state import QA, State


message_style = dict(display="inline-block", padding="1em", border_radius="8px", max_width=["30em", "30em", "50em", "50em", "50em", "50em"])


def message(qa: QA) -> rx.Component:
    """A single question/answer message.

    Args:
        qa: The question/answer pair.

    Returns:
        A component displaying the question/answer pair.
    """
    return rx.box(
        rx.box(
            rx.markdown(
                qa.question,
                background_color="#ffffff",
                color="#3E63DD",
                **message_style,
            ),
            text_align="right",
            margin_top="0.5em",
        ),
        rx.box(
            rx.markdown(
                qa.answer,
                background_color="#3E63DD",
                color="#ffffff",
                **message_style,
            ),
            text_align="left",
            padding_top="0.5em",
        ),
        width="100%",
    )


def chat() -> rx.Component:
    """List all the messages in a single conversation."""
    return rx.vstack(
        rx.box(rx.foreach(State.chats[State.current_chat], message), width="100%"),
        py="8",
        flex="1",
        width="100%",
        max_width="50em",
        padding_x="4px",
        align_self="center",
        overflow="hidden",
        padding_bottom="5em",
    )


def action_bar() -> rx.Component:
    """The action bar to send a new message."""
    return rx.center(
        rx.vstack(
            rx.chakra.form(
                rx.chakra.form_control(
                    rx.hstack(
                        rx.radix.text_field.root(
                            rx.radix.text_field.input(
                                placeholder="Type something...",
                                id="question",
                                width=["15em", "20em", "45em", "50em", "50em", "50em"],
                                background_color="#ffffff",
                                color="#3E63DD"
                            ),
                        ),
                        rx.button(
                            rx.cond(
                                State.processing,
                                loading_icon(height="1em"),
                                rx.text("Send"),
                            ),
                            type="submit",
                        ),
                        align_items="center",
                    ),
                    is_disabled=State.processing,
                ),
                on_submit=State.process_question,
                reset_on_submit=True,
            ),
            rx.text(
                "MRIBot may return factually incorrect or misleading responses. Use discretion.",
                text_align="center",
                font_size=".75em",
                color="#3E63DD"
            ),
            rx.hstack(
                rx.text("Built with", color="#ffffff"),
                rx.image(src="https://i.ibb.co/Df6px36/Screenshot-2024-04-20-at-1-53-45-PM.png", height="30px", width="auto"),
                align_items="flex-end",
                justify_content="space-between",
            ),
            align_items="center",
            justify_content="in-between",
        ),
        position="sticky",
        bottom="0",
        left="0",
        padding_y="16px",
        backdrop_filter="auto",
        backdrop_blur="lg",
        background_color="#A0B6FF",
        align_items="stretch",
        width="100%",
    )