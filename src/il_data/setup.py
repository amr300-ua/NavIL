from setuptools import find_packages, setup


package_name = "il_data"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test", "test.*")),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="NavIL team",
    maintainer_email="navil-team@example.com",
    description="Captura y preparación de datos para aprendizaje por imitación en NavIL.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "navil-record = il_data.recorder:main",
            "navil-bag-to-dataset = il_data.bag_to_dataset:main",
            "navil-generate-fake-dataset = il_data.generate_fake_dataset:main",
        ],
    },
)
