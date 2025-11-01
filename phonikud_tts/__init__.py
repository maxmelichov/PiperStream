from phonikud import phonemize

# Optional imports: allow VC-only mode to run without Piper/Phonikud ONNX libs
try:
    from piper_onnx import Piper  # type: ignore
except Exception:  # pragma: no cover
    class Piper:  # minimal stub to fail only when used
        def __init__(self, *a, **kw):
            raise ModuleNotFoundError(
                "piper_onnx is not installed. tts2vc mode requires Piper ONNX."
            )

try:
    from phonikud_onnx import Phonikud  # type: ignore
except Exception:  # pragma: no cover
    class Phonikud:  # minimal stub to fail only when used
        def __init__(self, *a, **kw):
            raise ModuleNotFoundError(
                "phonikud_onnx is not installed. tts2vc mode requires Phonikud ONNX."
            )