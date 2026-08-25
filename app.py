from flask import Flask, render_template, request, flash
from engine import run_full_evaluation, read_evaluations

app = Flask(__name__)
app.secret_key = "nea-local-dev-key"

@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    diagnostics = None

    if request.method == "POST":
        payload = {
            "ticker": request.form.get("ticker", ""),
            "option_type": request.form.get("option_type", "call"),
            "K": request.form.get("K", ""),
            "expiry_date": request.form.get("expiry_date", ""),
            "current_option_price": request.form.get("current_option_price", "0"),
            "risk_free_r_pct": request.form.get("risk_free_r_pct", "0.5"),
            "error_threshold_percent": request.form.get("error_threshold_percent", "10"),
        }

        try:
            results, diagnostics = run_full_evaluation(payload)
            flash("Evaluation completed and saved to evaluations.csv", "success")
        except Exception as e:
            flash(str(e), "error")

    return render_template("index.html", results=results, diagnostics=diagnostics)

@app.route("/history")
def history():
    df = read_evaluations()
    if df is None or df.empty:
        rows = []
        columns = []
    else:
        df = df.copy()
        df["risk_free_r"] = (df["risk_free_r"] * 100).round(3)
        df["volatility_used"] = (df["volatility_used"] * 100).round(3)
        df["weighted_price"] = df["weighted_price"].round(6)
        df["single_best_price"] = df["single_best_price"].round(6)
        rows = df.to_dict(orient="records")
        columns = list(df.columns)

    return render_template("history.html", rows=rows, columns=columns)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
