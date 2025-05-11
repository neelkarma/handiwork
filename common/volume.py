import subprocess


def set_volume(value):
    subprocess.run(["wpctl", "set-volume", "@DEFAULT_SINK@", str(value) + "%"])


def get_volume():
    return int(
        float(
            subprocess.run(
                ["wpctl", "get-volume", "@DEFAULT_SINK@"],
                capture_output=True,
                text=True,
            )
            .stdout.strip("\n")
            .split()[1]
        )
        * 100
    )
