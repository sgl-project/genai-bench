"""Click command support for integer lists on iteration options."""

import click


class IterationValuesCommand(click.Command):
    """Accept several values after a concurrency or batch-size option."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        options = {
            flag: param
            for param in self.params
            if isinstance(param, click.Option)
            for flag in (*param.opts, *param.secondary_opts)
        }
        expanded = []
        index = 0
        while index < len(args):
            token = args[index]
            expanded.append(token)
            index += 1
            if token == "--":
                expanded.extend(args[index:])
                break
            flag, separator, _ = token.partition("=")
            option = options.get(flag)
            if option is None or option.is_flag or option.count:
                continue
            if option.name in {"num_concurrency", "batch_size"}:
                first_value = not separator
                while index < len(args):
                    value = args[index]
                    if value.startswith("-") and not value[1:2].isdigit():
                        break
                    if not first_value:
                        expanded.append(flag)
                    expanded.append(value)
                    index += 1
                    first_value = False
            elif not separator:
                # Values belonging to other options can themselves look like flags.
                expanded.extend(args[index : index + option.nargs])
                index += option.nargs
        return super().parse_args(ctx, expanded)
