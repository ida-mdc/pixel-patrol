from dash import html
import dash_bootstrap_components as dbc


def create_info_icon(widget_id: str, help_text: str):
    """
    Creates a standardized 'i' icon that reveals a popover on hover/click.
    """
    if not help_text:
        return None

    target_id = f"info-target-{widget_id}"

    return html.Div(
        [
            html.I(
                className="bi bi-info-circle-fill",  # Bootstrap icon
                id=target_id,
                style={
                    "cursor": "pointer",
                    "color": "#6c757d",
                    "fontSize": "1.2rem",
                    "marginLeft": "10px"
                }
            ),
            dbc.Popover(
                [
                    dbc.PopoverHeader("Widget Info"),
                    dbc.PopoverBody(help_text),
                ],
                target=target_id,
                trigger="legacy",  # hover + click behavior
                placement="left",
            ),
        ],
        className="d-flex align-items-center"
    )


def create_widget_card(title: str, content: list, widget_id: str, help_text: str = None):
    """
    Wraps widget content in a standardized card with a header.

    Args:
        title: The display title of the widget.
        content: The list of Dash components (graphs, tables) to display.
        widget_id: Unique ID for mapping the info popover.
        help_text: Optional markdown/text to show in the info popover.
    """

    header_children = [
        html.H4(title, className="m-0 text-primary"),
    ]

    # If help text exists, add the icon to the right side of the header
    if help_text:
        # Use a flexible div to push the icon to the far right
        header_children.append(
            html.Div(create_info_icon(widget_id, help_text), className="ms-auto")
        )

    return dbc.Card(
        [
            dbc.CardHeader(
                html.Div(header_children, className="d-flex align-items-center w-100"),
                className="bg-light"
            ),
            dbc.CardBody(
                content,
                className="p-3"
            ),
        ],
        className="mb-4 shadow-sm",  # Standard margin bottom and shadow
        style={"borderRadius": "8px"}
    )