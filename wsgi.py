from waitress import serve
import app0

serve(app0, host = "0.0.0.0", port = 8888)