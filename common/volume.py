import subprocess


def set_volume(value):
    subprocess.run(["pamixer", "--set-volume", str(int(value * 100))])


def get_volume():
    return int(
        subprocess.run(
            ["pamixer", "--get-volume"], capture_output=True, text=True
        ).stdout.strip("\n")
    )
