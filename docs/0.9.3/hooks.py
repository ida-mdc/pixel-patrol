"""MkDocs hooks: fill generated sections of the docs from the code at build time."""

OPTIONS_TABLE = "<!-- process-args-table -->"


def _table(heading, rows):
    return "\n".join([f"**{heading}**", "", "| Name | Description |", "|---|---|", *rows])


def on_page_markdown(markdown, **kwargs):
    if OPTIONS_TABLE not in markdown:
        return markdown
    from pixel_patrol_base.process_args import PROCESS_ARGS  # needs pixel-patrol-base installed

    arguments, options = [], []
    for arg in PROCESS_ARGS:
        if arg.argument:
            arguments.append(f"| `{arg.name.upper()}` | {arg.full_help} |")
            continue
        flags = ", ".join(sorted(arg.cli_flags, key=lambda f: f.startswith("--")))  # short flag first
        label = f"{flags} {arg.metavar}" if arg.metavar else flags
        options.append(f"| `{label}` | {arg.full_help} |")
    return markdown.replace(OPTIONS_TABLE, _table("Arguments", arguments) + "\n\n" + _table("Options", options))
