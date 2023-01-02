#  Copyright (C) Richard Vogel 9.9.2022
#  Author Richard Vogel <richard.vogel@gmx.com>
#  All Rights Reserved
from __future__ import annotations
from . import split_rule

def get_split_rule_by_name(name: str) -> Type[SplitRule] | None:
    """ returns the split rule by name, if available
    will also return None if the found class is not a AbstractSplitRule subtype"""

    if hasattr(split_rule, name):
        maybe_sr = getattr(split_rule, name)
        if issubclass(maybe_sr, split_rule.AbstractSplitRule):
            return maybe_sr

    return None
