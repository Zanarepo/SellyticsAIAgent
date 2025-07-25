from flask import Flask, jsonify
from supabase import create_client, Client
from dotenv import load_dotenv
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta, timezone
import numpy as np
import logging
from apscheduler.schedulers.background import BackgroundScheduler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Initialize Supabase client
load_dotenv()
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
if not supabase_url or not supabase_key:
    logger.error("Missing SUPABASE_URL or SUPABASE_KEY")
    raise ValueError("Supabase credentials not found")
supabase: Client = create_client(supabase_url, supabase_key)

# Sales forecasting logic
def forecast_demand():
    try:
        sales = supabase.table("dynamic_sales").select("dynamic_product_id, store_id, quantity, sold_at").execute().data
        if not sales:
            logger.info("No sales data found")
            return []
        sales_df = pd.DataFrame(sales)
        sales_df["sold_at"] = pd.to_datetime(sales_df["sold_at"], utc=True)
        sales_df["month"] = sales_df["sold_at"].dt.to_period("M").astype(str)

        monthly_sales = sales_df.groupby(["dynamic_product_id", "store_id", "month"])["quantity"].sum().reset_index()
        forecasts = []
        for (product_id, store_id) in monthly_sales[["dynamic_product_id", "store_id"]].drop_duplicates().values:
            product_data = monthly_sales[(monthly_sales["dynamic_product_id"] == product_id) & (monthly_sales["store_id"] == store_id)].sort_values("month")
            if len(product_data) < 2:
                continue

            X = [[i] for i in range(len(product_data))]
            y = product_data["quantity"].values
            model = LinearRegression()
            model.fit(X, y)
            predicted_demand = max(0, model.predict([[len(product_data)]])[0])

            inventory = supabase.table("dynamic_inventory").select("available_qty, reorder_level").eq("dynamic_product_id", product_id).eq("store_id", store_id).execute().data
            product = supabase.table("dynamic_product").select("name").eq("id", product_id).execute().data
            store = supabase.table("stores").select("shop_name").eq("id", store_id).execute().data

            current_stock = int(inventory[0]["available_qty"]) if inventory else 0
            reorder_level = int(inventory[0]["reorder_level"]) if inventory and inventory[0]["reorder_level"] is not None else 0
            product_name = product[0]["name"] if product else f"Product ID: {product_id}"
            shop_name = store[0]["shop_name"] if store else f"Store ID: {store_id}"

            recommendation = "Restock recommended" if predicted_demand > current_stock + reorder_level else "No restock needed"

            forecast = {
                "dynamic_product_id": int(product_id),
                "store_id": int(store_id),
                "predicted_demand": round(float(predicted_demand), 2),
                "current_stock": current_stock,
                "product_name": product_name,
                "shop_name": shop_name,
                "recommendation": recommendation,
                "forecast_period": (datetime.now(timezone.utc) + timedelta(days=30)).strftime("%Y-%m"),
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            forecasts.append(forecast)

        if forecasts:
            logger.info(f"Inserting {len(forecasts)} forecasts into Supabase")
            supabase.table("forecasts").insert(forecasts).execute()
        return forecasts
    except Exception as e:
        logger.error(f"Error in forecast_demand: {str(e)}")
        raise

def detect_anomalies():
    try:
        sales = supabase.table("dynamic_sales").select("id, dynamic_product_id, store_id, quantity, sold_at").execute().data
        if not sales:
            logger.info("No sales data found for anomalies")
            return []
        sales_df = pd.DataFrame(sales)
        sales_df["sold_at"] = pd.to_datetime(sales_df["sold_at"], utc=True)

        anomalies = []
        for (product_id, store_id) in sales_df[["dynamic_product_id", "store_id"]].drop_duplicates().values:
            product_sales = sales_df[(sales_df["dynamic_product_id"] == product_id) & (sales_df["store_id"] == store_id)]
            quantities = product_sales["quantity"].values
            if len(quantities) < 3:
                continue

            mean = np.mean(quantities)
            std = np.std(quantities)
            z_scores = [(q - mean) / std if std > 0 else 0 for q in quantities]
            anomaly_indices = [i for i, z in enumerate(z_scores) if abs(z) > 3]

            for idx in anomaly_indices:
                row = product_sales.iloc[idx]
                anomalies.append({
                    "dynamic_product_id": int(row["dynamic_product_id"]),
                    "store_id": int(row["store_id"]),
                    "quantity": int(row["quantity"]),
                    "sold_at": row["sold_at"].isoformat(),
                    "anomaly_type": "High" if z_scores[idx] > 0 else "Low",
                    "created_at": datetime.now(timezone.utc).isoformat()
                })

        if anomalies:
            logger.info(f"Inserting {len(anomalies)} anomalies into Supabase")
            supabase.table("anomalies").insert(anomalies).execute()
        return anomalies
    except Exception as e:
        logger.error(f"Error in detect_anomalies: {str(e)}")
        raise

def sales_trends():
    try:
        sales = supabase.table("dynamic_sales").select("dynamic_product_id, store_id, quantity, sold_at").execute().data
        if not sales:
            logger.info("No sales data found for trends")
            return {}
        sales_df = pd.DataFrame(sales)
        sales_df["sold_at"] = pd.to_datetime(sales_df["sold_at"], utc=True)
        sales_df["month"] = sales_df["sold_at"].dt.to_period("M").astype(str)

        trends = {
            "top_products": sales_df.groupby("dynamic_product_id")["quantity"].sum().nlargest(5).to_dict(),
            "top_stores": sales_df.groupby("store_id")["quantity"].sum().nlargest(5).to_dict(),
            "monthly_trends": sales_df.groupby("month")["quantity"].sum().to_dict()
        }
        return trends
    except Exception as e:
        logger.error(f"Error in sales_trends: {str(e)}")
        raise

def process_inquiry(inquiry_text):
    try:
        inquiry_text = inquiry_text.lower()
        responses = {
            "stock": "Please check the inventory dashboard or contact support for stock details.",
            "availability": "Availability can be checked in real-time on the platform.",
            "order": "Orders can be placed through the platform or by contacting support.",
            "delivery": "Delivery timelines depend on your location. Please provide more details.",
            "price": "Pricing details are available in the product catalog."
        }
        for key, response in responses.items():
            if key in inquiry_text:
                return response
        return "Thank you for your inquiry. Please provide more details or contact support."
    except Exception as e:
        logger.error(f"Error in process_inquiry: {str(e)}")
        raise

def handle_inquiries():
    try:
        inquiries = supabase.table("customer_inquiries").select("id, inquiry_text").eq("status", "pending").execute().data
        logger.info(f"Found {len(inquiries)} pending inquiries")
        for inquiry in inquiries:
            response = process_inquiry(inquiry["inquiry_text"])
            logger.info(f"Processing inquiry {inquiry['id']}: {inquiry['inquiry_text']} -> {response}")
            supabase.table("customer_inquiries").update({
                "response_text": response,
                "status": "responded",
                "created_at": datetime.now(timezone.utc).isoformat()
            }).eq("id", inquiry["id"]).execute()
        return inquiries
    except Exception as e:
        logger.error(f"Error in handle_inquiries: {str(e)}")
        raise

# Scheduled task to run analysis automatically
def run_analysis():
    try:
        logger.info("Running scheduled analysis...")
        forecast_demand()
        detect_anomalies()
        sales_trends()
        handle_inquiries()
        logger.info("Scheduled analysis completed")
    except Exception as e:
        logger.error(f"Error in scheduled analysis: {str(e)}")

# Set up scheduler
from datetime import datetime, timezone
from apscheduler.schedulers.background import BackgroundScheduler

scheduler = BackgroundScheduler()

# Schedule the job to run every 5 minutes, starting now
scheduler.add_job(run_analysis, 'interval', weeks=1, next_run_time=datetime.now(timezone.utc))

scheduler.start()


@app.route('/forecast', methods=['GET'])
def forecast_endpoint():
    try:
        forecasts = forecast_demand()
        anomalies = detect_anomalies()
        trends = sales_trends()
        return jsonify({
            "forecasts": forecasts,
            "anomalies": anomalies,
            "trends": trends
        }), 200
    except Exception as e:
        logger.error(f"Error in /forecast: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/inquiries', methods=['GET'])
def inquiries_endpoint():
    try:
        inquiries = handle_inquiries()
        return jsonify({"inquiries": inquiries}), 200
    except Exception as e:
        logger.error(f"Error in /inquiries: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def root_endpoint():
    return jsonify({"message": "Sellytics AI Agent Backend"}), 200

if __name__ == "__main__":
    try:
        app.run(host='0.0.0.0', port=int(os.getenv("PORT", 5000)))
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()