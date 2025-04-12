import pandas as pd
import os
import streamlit as st
from filelock import FileLock

items = pd.read_csv("crude_csv/items.csv")
transaction_items = pd.read_csv("crude_csv/transaction_items.csv")
transaction_items.drop(columns=["Unnamed: 0"],inplace=True)
merchant = pd.read_csv("crude_csv/merchant.csv")

def df_merger(merchant_name, tr_data, tr_items):
    merchant_row = merchant[merchant["merchant_name"] == merchant_name]
    if len(merchant_row) == 0:
        print("Merchant name does not exist")

    merchant_id = merchant_row.iloc[0]["merchant_id"]
    df_items = items[items["merchant_id"] == merchant_id]

    df_merged_transactions = pd.merge(tr_data, tr_items, on="order_id", how="inner")
    df_merged = pd.merge(df_merged_transactions, df_items, on="item_id", how="inner")
    df_merged.drop_duplicates(inplace=True)

    return df_merged

def merged_df(merchant_name, time, live=False, all=False):
    if live:
        path = f"simulated_stream_{merchant_name}.csv"
        lock = FileLock(f"{path}.lock")
        if not os.path.exists(path):
            st.warning(f"No real-time data yet for {merchant_name}")
            return pd.DataFrame()
        with lock:
            tr_data = pd.read_csv(path)

        tr_data["order_time"] = pd.to_datetime(tr_data["order_time"])
        if merchant_name == 'Bagel Bros':
            tr_items = pd.read_csv('processed_csv/transaction_bagel_items.csv')
        else:
            tr_items = pd.read_csv('processed_csv/transaction_noodle_items.csv')

        df_merged = df_merger(merchant_name, tr_data, tr_items)

        return df_merged

    # Static fallback (non-live)
    det = "" if all is False else "_all"

    if merchant_name == 'Bagel Bros':
        tr_data = pd.read_csv(f'processed_csv/transaction_bagel_data{det}.csv')
        tr_items = pd.read_csv('processed_csv/transaction_bagel_items.csv')
    else:
        tr_data = pd.read_csv(f'processed_csv/transaction_noodle_data{det}.csv')
        tr_items = pd.read_csv('processed_csv/transaction_noodle_items.csv')

    tr_data["order_time"] = pd.to_datetime(tr_data["order_time"])

    if time == 'This Week':
        tr_data = tr_data[
            (tr_data['order_time'] > pd.to_datetime('2023-06-11')) &
            (tr_data['order_time'] < pd.to_datetime('2023-06-17'))
                ]
    elif time == 'This Month':
        tr_data = tr_data[tr_data['order_time'] < pd.to_datetime('2023-06-17')]
    else:
        pass

    df_merged = df_merger(merchant_name, tr_data, tr_items)

    return df_merged

def most_ordered_product(merchant_name, time):
    df_merged = merged_df(merchant_name, None, live=True)

    if time == 'This Week':
        df_week = merged_df(merchant_name, time)
        df = pd.concat([df_week, df_merged], ignore_index=True)
    elif time == "This Month":
        df_month = merged_df(merchant_name, time)
        df = pd.concat([df_month, df_merged], ignore_index=True)
    else:
        df = df_merged

    df_item_counts = df.groupby('item_name').size().reset_index(name = "Total orders")
    return df_item_counts

def order_per_hour(merchant_name, time):
    df_merged = merged_df(merchant_name, time)

    df_merged['order_hour'] = df_merged['order_time'].dt.hour
    df_order_per_hour = df_merged.groupby('order_hour').size().reset_index(name="Total orders")

    return df_order_per_hour

def order_per_date(merchant_name, time):
    df_merged = merged_df(merchant_name, None, live=True)

    df_merged['order_hour'] = df_merged['order_time'].dt.hour
    df_merged['order_date'] = df_merged['order_time'].dt.date

    df_order_per_date = df_merged.groupby(['order_date', 'order_hour']).size().reset_index(name="Total orders")
    df_order_per_date['hour_name'] = df_order_per_date['order_hour'].apply(lambda x: f"{x:02d}:00")


    #df_merged['order_date'] = pd.to_datetime(df_merged['order_time']).dt.hour
    #df_order_per_date = df_merged.groupby('order_date').size().reset_index(name="Total orders")
    #df_order_per_date['hour_name'] = df_order_per_date['order_date'].apply(lambda x: f"{x:02d}:00")

    if time == 'This Week':
        df_week = merged_df(merchant_name, time)
        df_week['order_hour'] = df_week['order_time'].dt.hour
        df_week['order_date'] = df_week['order_time'].dt.date

        df_order_week = df_week.groupby(['order_date', 'order_hour']).size().reset_index(name="Total orders")
        df_order_week['hour_name'] = df_order_week['order_hour'].apply(lambda x: f"{x:02d}:00")

        df_final = pd.concat([df_order_week, df_order_per_date], ignore_index=True)
    elif time == 'This Month':
        df_month = merged_df(merchant_name, time)        
        df_month['order_hour'] = df_month['order_time'].dt.hour
        df_month['order_date'] = df_month['order_time'].dt.date

        df_order_month = df_month.groupby(['order_date', 'order_hour']).size().reset_index(name="Total orders")
        df_order_month['hour_name'] = df_order_month['order_hour'].apply(lambda x: f"{x:02d}:00")

        df_final = pd.concat([df_order_month, df_order_per_date], ignore_index=True)
    else:
        df_final = df_order_per_date
    
    df_final['datetime'] = df_final['order_date'].astype(str) + " " + df_final['hour_name']
    
    return df_final

def revenue_per_date(merchant_name, time):
    df_merged = merged_df(merchant_name, None, live=True)

    df_merged['order_hour'] = df_merged['order_time'].dt.hour
    df_merged['order_date'] = df_merged['order_time'].dt.date

    df_revenue_per_date = df_merged.groupby(['order_date', 'order_hour'])['item_price'].sum().reset_index(name="Total Revenue (RM)")
    df_revenue_per_date['order_hour'] = df_revenue_per_date['order_hour'].apply(lambda x: f"{x:02d}:00")

    if time == 'This Week':
        df_week = merged_df(merchant_name, time)
        df_week['order_hour'] = df_week['order_time'].dt.hour
        df_week['order_date'] = df_week['order_time'].dt.date

        df_revenue_week = df_week.groupby(['order_date', 'order_hour'])['item_price'].sum().reset_index(name="Total Revenue (RM)")
        df_revenue_week['order_hour'] = df_revenue_week['order_hour'].apply(lambda x: f"{x:02d}:00")

        df_final = pd.concat([df_revenue_week, df_revenue_per_date], ignore_index=True)
    elif time == 'This Month':
        df_month = merged_df(merchant_name, time)        
        df_month['order_hour'] = df_month['order_time'].dt.hour
        df_month['order_date'] = df_month['order_time'].dt.date

        df_revenue_month = df_month.groupby(['order_date', 'order_hour'])['item_price'].sum().reset_index(name="Total Revenue (RM)")
        df_revenue_month['order_hour'] = df_revenue_month['order_hour'].apply(lambda x: f"{x:02d}:00")

        df_final = pd.concat([df_revenue_month, df_revenue_per_date], ignore_index=True)
    else:
        df_final = df_revenue_per_date
    
    df_final['datetime'] = df_final['order_date'].astype(str) + " " + df_final['order_hour']

    #if time == 'Today':
    #    df_merged['order_date'] = pd.to_datetime(df_merged['order_time']).dt.hour
    #    df_revenue_per_date = df_merged.groupby('order_date')['item_price'].sum().reset_index(name="Total Revenue (RM)")
    #    df_revenue_per_date['hour_name'] = df_revenue_per_date['order_date'].apply(lambda x: f"{x:02d}:00")
    #elif time == 'This Week':
    #    df_merged['order_date'] = pd.to_datetime(df_merged['order_time']).dt.day
    #    df_revenue_per_date = df_merged.groupby('order_date')['item_price'].sum().reset_index(name="Total Revenue (RM)")
    #    df_revenue_per_date['day_name'] = df_revenue_per_date['order_date'].apply(lambda x: pd.to_datetime(x).strftime('%A'))
    #else:
    #    df_merged['order_date'] = pd.to_datetime(df_merged['order_time'])

        # Get the first day of the month for each order
    #    df_merged['month_start'] = df_merged['order_date'].values.astype('datetime64[M]')

        # Compute how many 7-day blocks (weeks) since the 1st of that month
    #    df_merged['week_from_1st'] = ((df_merged['order_date'] - df_merged['month_start']).dt.days // 7) + 1

        # Optional: label as something like "2023-06 Week 2"
    #    df_merged['order_date'] = df_merged['month_start'].dt.strftime('%Y-%m') + ' Week ' + df_merged['week_from_1st'].astype(str)

        # Group by that label
    #    df_revenue_per_date = df_merged.groupby('order_date')['item_price'].sum().reset_index()
    #    df_revenue_per_date.rename(columns={'item_price': 'Total Revenue (RM)'}, inplace=True)
    

    return df_final


def orders_and_revenue_per_date(merchant_name, time):
    df_merged = merged_df(merchant_name, time)

    # Convert order_time to date
    df_merged['order_date'] = df_merged['order_time'].dt.date

    # Group by date to get total orders and total revenue
    df_summary = df_merged.groupby('order_date').agg(
        **{
            "Total orders": ("order_id", "nunique"),
            "Total Revenue (RM)": ("item_price", "sum")
        }
    ).reset_index()

    # Add day name
    df_summary['day_name'] = df_summary['order_date'].apply(lambda x: x.strftime('%A'))

    return df_summary


def total_revenue(merchant_name, time, live=False):
    df_merged = merged_df(merchant_name, live=live, time=time)
    if df_merged.empty:
        return 0

    base_rev = df_merged["item_price"].sum()

    if time == "Today":
        past_rev = 0
    elif time == 'This Week':
        df_merged_week = merged_df(merchant_name, time=time)
        past_rev = df_merged_week["item_price"].sum()
    else:
        df_merged_month = merged_df(merchant_name, time=time)
        past_rev = df_merged_month["item_price"].sum()
        
    return base_rev + past_rev

def total_orders(merchant_name, time, live=False) :
    df_merged = merged_df(merchant_name, live=live, time=time)
    if df_merged.empty:
        return 0

    base_orders = df_merged["order_id"].size

    if time == "Today":
        past_orders = 0
    elif time == 'This Week':
        df_merged_week = merged_df(merchant_name, time=time)
        past_orders = df_merged_week["order_id"].size
    else:
        df_merged_month = merged_df(merchant_name, time=time)
        past_orders = df_merged_month["order_id"].size
        
    return base_orders + past_orders

    #df_merged = merged_df(merchant_name, time)

    #return len(df_merged["order_id"].unique())

def average_driver_waiting_time(merchant_name, time):
    df_merged = merged_df(merchant_name, time)

    df_merged['order_hour'] = df_merged['order_time'].dt.hour
    df_avg_waiting =  df_merged.groupby('order_hour')['driver_waiting_time'].mean().reset_index(name="Driver Waiting Time (minutes)")

    return df_avg_waiting

def average_meal_ready_time(merchant_name, time):
    df_merged = merged_df(merchant_name, time)

    df_merged['order_hour'] = df_merged['order_time'].dt.hour
    df_avg_ready =  df_merged.groupby('order_ready')['driver_waiting_time'].mean().reset_index(name="Time to prepare order (minutes)")

    return df_avg_ready

def avg_wait_and_ready_time(merchant_name):
    df_merged = merged_df(merchant_name, None, live=True)

    # Create hour column for grouping
    df_merged["order_time"] = pd.to_datetime(df_merged['order_time']).dt.hour

    last_hour = df_merged['order_time'].tail(1).values[0]
    hour = last_hour - 1 if last_hour != 22 else last_hour

    idx = df_merged[df_merged['order_time'] == hour].index.min()
    driver_wait = df_merged['driver_waiting_time'].iloc[:idx+1].mean()
    order_prep = df_merged['order_ready'].iloc[:idx+1].mean()

    # Group by hour and calculate both metrics
    #df_summary = df_merged.groupby("order_hour").agg({
    #    "driver_waiting_time": "mean",
    #    "order_ready": "mean"
    #}).reset_index()

    # Rename columns
    #df_summary.rename(columns={
    #    "driver_waiting_time": "Driver Waiting Time (minutes)",
    #    "order_ready": "Time to prepare order (minutes)"
    #}, inplace=True)

    return driver_wait, order_prep
