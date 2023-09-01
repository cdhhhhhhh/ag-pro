custom_imports = dict(
    imports=['lib.hooks'],
    allow_failed_imports=False)


custom_hooks = [
    dict(type='MySelfExpHook', interval=50)
]
