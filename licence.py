import os

# Define the LICENSE_TEXT variable (replace this with your license text)
LICENSE_TEXT = """\
# Identifiability Guarantees For Time Series Representation via Contrastive Sparsity-inducing
# Copyright 2024, ICLR 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""


def add_license_to_file(file_path):
    with open(file_path, "r+") as file:
        content = file.read()
        if LICENSE_TEXT.strip() not in content:  # Avoid duplicating the license
            file.seek(0)
            file.write(LICENSE_TEXT + "\n" + content)


def process_directory(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                add_license_to_file(os.path.join(root, file))


# Set the target directory (replace with your target directory)
target_directory = "./"  # Current directory or specify your project directory
process_directory(target_directory)
