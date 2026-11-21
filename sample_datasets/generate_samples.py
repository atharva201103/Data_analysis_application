"""
Generates realistic sample datasets for DataSense AI.
Includes Retail Sales, HR Analytics, and E-commerce Orders.
Each dataset includes realistic data quality anomalies (missing values, duplicates,
formatting inconsistencies, outliers, etc.) for testing.
"""

import os
import pandas as pd
import numpy as np


def generate_retail_sales() -> pd.DataFrame:
    np.random.seed(42)
    n = 250
    dates = pd.date_range(start="2024-01-01", periods=180, freq="D")
    date_strings = [d.strftime("%Y-%m-%d") for d in dates]
    
    categories = ["Electronics", "Home Appliances", "Office Supplies", "Fashion", "Sports"]
    category_messy = ["Electronics", "electronics ", "Electronics", "Home Appliances", "home appliances", 
                      "Office Supplies", "Office Supplies ", "Fashion", "Sports", "sports"]
    regions = ["North", "South", "East", "West", "Central"]
    payment_methods = ["Credit Card", "PayPal", "Bank Transfer", "Debit Card"]
    
    data = []
    for i in range(1, n + 1):
        cat = np.random.choice(category_messy)
        clean_cat = cat.strip().title()
        
        if clean_cat == "Electronics":
            unit_price = np.random.uniform(150, 1200)
            qty = np.random.randint(1, 8)
            cost_factor = 0.65
        elif clean_cat == "Home Appliances":
            unit_price = np.random.uniform(80, 600)
            qty = np.random.randint(1, 10)
            cost_factor = 0.70
        elif clean_cat == "Fashion":
            unit_price = np.random.uniform(25, 200)
            qty = np.random.randint(1, 15)
            cost_factor = 0.50
        elif clean_cat == "Office Supplies":
            unit_price = np.random.uniform(10, 95)
            qty = np.random.randint(2, 30)
            cost_factor = 0.55
        else:
            unit_price = np.random.uniform(20, 350)
            qty = np.random.randint(1, 12)
            cost_factor = 0.60
            
        discount = np.random.choice([0.0, 0.05, 0.10, 0.15, 0.20], p=[0.4, 0.25, 0.15, 0.15, 0.05])
        gross_sales = round(unit_price * qty, 2)
        net_sales = round(gross_sales * (1 - discount), 2)
        total_cost = round(unit_price * qty * cost_factor, 2)
        profit = round(net_sales - total_cost, 2)
        
        row = {
            "Transaction_ID": f"TXN-{1000 + i}",
            "Date": np.random.choice(date_strings),
            "Customer_ID": f"CUST-{np.random.randint(101, 160)}",
            "Category": cat,
            "Region": np.random.choice(regions),
            "Quantity": qty,
            "Unit_Price": unit_price,
            "Discount": f"{int(discount * 100)}%" if i % 5 == 0 else discount, # messy type
            "Total_Revenue": net_sales,
            "Total_Cost": total_cost,
            "Profit": profit,
            "Payment_Method": np.random.choice(payment_methods)
        }
        data.append(row)
        
    df = pd.DataFrame(data)
    
    # Introduce deliberate missing values
    df.loc[df.sample(15, random_state=1).index, "Profit"] = np.nan
    df.loc[df.sample(8, random_state=2).index, "Customer_ID"] = np.nan
    df.loc[df.sample(6, random_state=3).index, "Payment_Method"] = np.nan
    
    # Introduce deliberate duplicate rows
    dup_rows = df.iloc[:4].copy()
    df = pd.concat([df, dup_rows], ignore_index=True)
    
    # Introduce outliers
    df.loc[12, "Total_Revenue"] = 85000.0  # Massive outlier sale
    df.loc[12, "Profit"] = 42000.0
    df.loc[34, "Quantity"] = 500           # Massive quantity
    df.loc[55, "Profit"] = -4500.0         # Heavy loss transaction
    
    return df


def generate_hr_analytics() -> pd.DataFrame:
    np.random.seed(123)
    n = 200
    departments = ["Engineering", "engineering ", "Sales", "sales", "Marketing", "HR", "Finance", "Product"]
    roles = ["Software Engineer", "Sales Executive", "Marketing Specialist", "HR Generalist", 
             "Financial Analyst", "Product Manager", "Tech Lead", "Senior Analyst"]
    
    data = []
    for i in range(1, n + 1):
        dept = np.random.choice(departments)
        clean_dept = dept.strip().capitalize()
        
        base_salary = {
            "Engineering": 115000,
            "Sales": 82000,
            "Marketing": 78000,
            "Hr": 68000,
            "Finance": 92000,
            "Product": 110000
        }.get(clean_dept, 75000)
        
        salary = base_salary + np.random.normal(0, 15000)
        perf_score = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.15, 0.50, 0.20, 0.10])
        years = np.random.choice(range(1, 15))
        overtime = np.random.choice(["Yes", "No"], p=[0.3, 0.7])
        
        # Attrition logic
        attrition_prob = 0.12
        if overtime == "Yes":
            attrition_prob += 0.15
        if perf_score <= 2:
            attrition_prob += 0.20
        if years < 2:
            attrition_prob += 0.10
        attrition = "Yes" if np.random.rand() < attrition_prob else "No"
        
        satisfaction = max(1, min(10, int(np.random.normal(7, 2))))
        absences = max(0, int(np.random.poisson(4)))
        
        row = {
            "Employee_ID": f"EMP-{2000 + i}",
            "Department": dept,
            "Role": np.random.choice(roles),
            "Hire_Date": pd.to_datetime("2020-01-01") + pd.to_timedelta(np.random.randint(0, 1400), unit='D'),
            "Monthly_Salary": round(salary, 2),
            "Performance_Rating": perf_score,
            "Years_at_Company": years,
            "Overtime": overtime,
            "Attrition": attrition,
            "Job_Satisfaction": satisfaction,
            "Annual_Absences": absences
        }
        data.append(row)
        
    df = pd.DataFrame(data)
    
    # Introduce data quality issues
    df.loc[df.sample(12, random_state=4).index, "Monthly_Salary"] = np.nan
    df.loc[df.sample(7, random_state=5).index, "Job_Satisfaction"] = np.nan
    
    # Duplicates
    df = pd.concat([df, df.iloc[:3].copy()], ignore_index=True)
    
    # Outliers
    df.loc[15, "Monthly_Salary"] = 380000.0  # Executive outlier
    df.loc[45, "Annual_Absences"] = 85        # High absence outlier
    
    return df


def generate_ecommerce_orders() -> pd.DataFrame:
    np.random.seed(999)
    n = 280
    dates = pd.date_range(start="2024-03-01", periods=120, freq="D")
    
    segments = ["Consumer", "Corporate", "Home Office"]
    categories = ["Technology", "Furniture", "Office Supplies"]
    subcats = {
        "Technology": ["Phones", "Accessories", "Machines", "Copiers"],
        "Furniture": ["Chairs", "Tables", "Bookcases", "Furnishings"],
        "Office Supplies": ["Storage", "Binders", "Paper", "Art", "Appliances"]
    }
    priorities = ["Critical", "High", "Medium", "Low"]
    
    data = []
    for i in range(1, n + 1):
        rand_offset = int(np.random.randint(0, 120))
        order_date = pd.to_datetime("2024-03-01") + pd.to_timedelta(rand_offset, unit='D')
        ship_date = order_date + pd.to_timedelta(int(np.random.randint(1, 7)), unit='D')
        cat = np.random.choice(categories)
        sub = np.random.choice(subcats[cat])
        
        sales = round(np.random.exponential(180) + 15, 2)
        qty = np.random.randint(1, 9)
        discount = np.random.choice([0.0, 0.1, 0.2, 0.3], p=[0.5, 0.25, 0.15, 0.1])
        margin = np.random.uniform(0.08, 0.38) if cat != "Furniture" else np.random.uniform(-0.15, 0.20)
        profit = round(sales * margin, 2)
        
        row = {
            "Order_ID": f"ORD-2024-{3000 + i}",
            "Order_Date": order_date.strftime("%Y-%m-%d"),
            "Ship_Date": ship_date.strftime("%Y-%m-%d"),
            "Customer_Segment": np.random.choice(segments),
            "Category": cat,
            "Sub_Category": sub,
            "Sales_Amount": sales,
            "Order_Quantity": qty,
            "Discount_Rate": discount,
            "Profit_Amount": profit,
            "Shipping_Cost": round(np.random.uniform(5, 45), 2),
            "Order_Priority": np.random.choice(priorities)
        }
        data.append(row)
        
    df = pd.DataFrame(data)
    
    # Missing values
    df.loc[df.sample(10, random_state=6).index, "Profit_Amount"] = np.nan
    df.loc[df.sample(8, random_state=7).index, "Shipping_Cost"] = np.nan
    
    # Duplicates
    df = pd.concat([df, df.iloc[:3].copy()], ignore_index=True)
    
    return df


def generate_marketing_analytics() -> pd.DataFrame:
    np.random.seed(777)
    n = 220
    channels = ["Google Search", "Meta Ads", "LinkedIn", "Email Newsletter", "YouTube Video", "TikTok", "meta ads", "Google Search "]
    audiences = ["B2B Executives", "Young Adults", "Tech Enthusiasts", "Homeowners", "Gamers"]
    
    data = []
    for i in range(1, n + 1):
        ch = np.random.choice(channels)
        clean_ch = ch.strip().title()
        
        if "Search" in clean_ch:
            spend = np.random.uniform(500, 8000)
            cpc = np.random.uniform(1.5, 4.5)
            conv_rate = np.random.uniform(0.02, 0.08)
        elif "Meta" in clean_ch or "Tiktok" in clean_ch:
            spend = np.random.uniform(300, 6000)
            cpc = np.random.uniform(0.5, 2.0)
            conv_rate = np.random.uniform(0.015, 0.05)
        elif "Linkedin" in clean_ch:
            spend = np.random.uniform(1000, 10000)
            cpc = np.random.uniform(5.0, 14.0)
            conv_rate = np.random.uniform(0.03, 0.10)
        else:
            spend = np.random.uniform(200, 2500)
            cpc = np.random.uniform(0.2, 1.2)
            conv_rate = np.random.uniform(0.04, 0.12)
            
        clicks = int(spend / cpc)
        impressions = int(clicks * np.random.uniform(15, 60))
        conversions = int(clicks * conv_rate)
        avg_order_value = np.random.uniform(80, 450)
        revenue = round(conversions * avg_order_value, 2)
        roas = round(revenue / spend, 2) if spend > 0 else 0.0
        
        row = {
            "Campaign_ID": f"CMP-{4000 + i}",
            "Campaign_Name": f"Q{np.random.choice([1,2,3,4])}_{clean_ch}_{i}",
            "Channel": ch,
            "Target_Audience": np.random.choice(audiences),
            "Spend": f"${spend:,.2f}" if i % 6 == 0 else round(spend, 2), # messy formatted string
            "Impressions": impressions,
            "Clicks": clicks,
            "Conversions": conversions,
            "Conversion_Rate": round(conv_rate * 100, 2),
            "Revenue": revenue,
            "ROAS": roas
        }
        data.append(row)
        
    df = pd.DataFrame(data)
    # Missing values
    df.loc[df.sample(12, random_state=8).index, "Revenue"] = np.nan
    df.loc[df.sample(9, random_state=9).index, "Conversions"] = np.nan
    
    # Duplicates & Outliers
    df = pd.concat([df, df.iloc[:3].copy()], ignore_index=True)
    df.loc[25, "Revenue"] = 145000.0 # High performer outlier
    return df


def generate_inventory_analytics() -> pd.DataFrame:
    np.random.seed(888)
    n = 200
    warehouses = ["Warehouse North", "Warehouse South", "Warehouse East", "Warehouse West", "warehouse north "]
    categories = ["Electronics", "Apparel", "Hardware", "Packaging", "Office"]
    
    data = []
    for i in range(1, n + 1):
        cat = np.random.choice(categories)
        wh = np.random.choice(warehouses)
        
        current_stock = np.random.randint(10, 1500)
        reorder_point = np.random.randint(50, 400)
        lead_time = np.random.randint(2, 28)
        unit_cost = round(np.random.uniform(5, 350), 2)
        total_valuation = round(current_stock * unit_cost, 2)
        holding_cost = round(total_valuation * 0.15, 2)
        
        if current_stock < reorder_point:
            risk = "High"
        elif current_stock < reorder_point * 1.5:
            risk = "Medium"
        else:
            risk = "Low"
            
        row = {
            "SKU_ID": f"SKU-{5000 + i}",
            "Product_Category": cat,
            "Warehouse": wh,
            "Current_Stock": current_stock,
            "Reorder_Point": reorder_point,
            "Lead_Time_Days": lead_time,
            "Unit_Cost": unit_cost,
            "Inventory_Valuation": total_valuation,
            "Annual_Holding_Cost": holding_cost,
            "Stockout_Risk": risk
        }
        data.append(row)
        
    df = pd.DataFrame(data)
    # Missing values
    df.loc[df.sample(8, random_state=10).index, "Lead_Time_Days"] = np.nan
    df.loc[df.sample(5, random_state=11).index, "Inventory_Valuation"] = np.nan
    
    # Duplicates & Anomalies
    df = pd.concat([df, df.iloc[:2].copy()], ignore_index=True)
    df.loc[10, "Current_Stock"] = -25 # Negative stock anomaly
    return df


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    
    retail_df = generate_retail_sales()
    retail_path = os.path.join(out_dir, "retail_sales_sample.csv")
    retail_df.to_csv(retail_path, index=False)
    print(f"Generated: {retail_path} ({len(retail_df)} rows, {len(retail_df.columns)} cols)")
    
    hr_df = generate_hr_analytics()
    hr_path = os.path.join(out_dir, "hr_analytics_sample.csv")
    hr_df.to_csv(hr_path, index=False)
    print(f"Generated: {hr_path} ({len(hr_df)} rows, {len(hr_df.columns)} cols)")
    
    ecom_df = generate_ecommerce_orders()
    ecom_path = os.path.join(out_dir, "ecommerce_orders_sample.csv")
    ecom_df.to_csv(ecom_path, index=False)
    print(f"Generated: {ecom_path} ({len(ecom_df)} rows, {len(ecom_df.columns)} cols)")

    mkt_df = generate_marketing_analytics()
    mkt_path = os.path.join(out_dir, "marketing_analytics_sample.csv")
    mkt_df.to_csv(mkt_path, index=False)
    print(f"Generated: {mkt_path} ({len(mkt_df)} rows, {len(mkt_df.columns)} cols)")

    inv_df = generate_inventory_analytics()
    inv_path = os.path.join(out_dir, "inventory_analytics_sample.csv")
    inv_df.to_csv(inv_path, index=False)
    print(f"Generated: {inv_path} ({len(inv_df)} rows, {len(inv_df.columns)} cols)")
    
    # Multi-sheet Excel file for testing worksheet selection
    excel_path = os.path.join(out_dir, "business_analytics_multisheet.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        retail_df.to_excel(writer, sheet_name="Retail_Sales", index=False)
        hr_df.to_excel(writer, sheet_name="HR_Workforce", index=False)
        ecom_df.to_excel(writer, sheet_name="Ecommerce_Orders", index=False)
        mkt_df.to_excel(writer, sheet_name="Marketing_ROI", index=False)
        inv_df.to_excel(writer, sheet_name="Supply_Chain", index=False)
    print(f"Generated multi-sheet Excel: {excel_path}")
