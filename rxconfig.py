import reflex as rx

config = rx.Config(
    app_name="mcpmultiagentsresearch",
    plugins=[
        rx.plugins.SitemapPlugin(),
        rx.plugins.TailwindV4Plugin(),
        rx.plugins.RadixThemesPlugin(),
    ]
)