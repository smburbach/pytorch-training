#!/usr/bin/python
# filename: base_config.py

#
# Copyright (c) 2024 Bryan Briney
# License: GNU General Public License, version 3.0 (http://opensource.org/licenses/gpl-3-0/)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

import json
from typing import Optional


class BaseConfig:
    def to_json(self, output: Optional[str] = None):
        """
        Save the config to a JSON file.

        Parameters
        ----------
        output : str, optional
            The path to the JSON file to save the config to. If None, the config is returned as a JSON string.
        """
        json_string = json.dumps(self.__dict__)
        if output is not None:
            with open(output, "w") as f:
                f.write(json_string)
        else:
            return json_string

    @classmethod
    def from_json(cls, json_path: str):
        with open(json_path, "r") as f:
            json_string = f.read()
        return cls(**json.loads(json_string))
