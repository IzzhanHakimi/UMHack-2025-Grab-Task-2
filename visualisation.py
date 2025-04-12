import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import textwrap

def prepare_data(shop, time):
    if shop == 'Bagel Bros':
        tr_data = pd.read_csv('processed_csv/transaction_bagel_data.csv')
        tr_items = pd.read_csv('processed_csv/transaction_bagel_items.csv')
    else:
        tr_data = pd.read_csv('processed_csv/transaction_noodle_data.csv')
        tr_items = pd.read_csv('processed_csv/transaction_noodle_items.csv')

    tr_data['order_time'] = pd.to_datetime(tr_data['order_time'])
    tr_data['driver_arrival_time'] = pd.to_datetime(tr_data['driver_arrival_time'])
    tr_data['driver_pickup_time'] = pd.to_datetime(tr_data['driver_pickup_time'])
    tr_data['delivery_time'] = pd.to_datetime(tr_data['delivery_time'])

    if time == 'Today':
        tr_data = tr_data[tr_data['order_time'] > pd.to_datetime('2023-06-17')]
    elif time == 'This Week':
        tr_data = tr_data[(tr_data['order_time'] >= pd.to_datetime('2023-06-11')) & (tr_data['order_time'] <= pd.to_datetime('2023-06-16'))]
    else:
        tr_data = tr_data[tr_data['order_time'] < pd.to_datetime('2023-06-17')]

    return tr_data, tr_items

def real_time_sim(tr_data, current):
    # Split data
    baseline = tr_data[tr_data['order_time'] <= current]
    upcoming = tr_data[tr_data['order_time'] > current]

    return baseline, upcoming  # return data left to stream

def sales_trend(revenue_record, time):
    sns.set(style="whitegrid")
    plt.figure(figsize=(5, 1))
    sns.lineplot(data=revenue_record, x="datetime", y="Total Revenue (RM)")

    ax = plt.gca()

    #if time == 'Today':
    #    label = pd.to_datetime(revenue_record['order_date']).dt.strftime('%H:%M')
    #elif time == 'This Week':
    ##    label = list(set(pd.to_datetime(revenue_record['order_date']).dt.strftime('%d-%m')))
    #else:
    #    label = ['Week 1', 'Week 2', 'Week 3']
    
    #ax.set_xticks(range(len(label)))
    ax.set_xticklabels([])
    ax.set_ylabel('Total Rev')
    ax.set_xlabel(f'Revenue Trend of {time}')

    return plt

def orders_trend(revenue_record, time):
    sns.set(style="whitegrid")
    plt.figure(figsize=(5, 1))
    sns.lineplot(data=revenue_record, x="datetime", y="Total orders")

    ax = plt.gca()

    #if time == 'Today':
    #    label = pd.to_datetime(revenue_record['order_date']).dt.strftime('%H:%M')
    #elif time == 'This Week':
    ##    label = list(set(pd.to_datetime(revenue_record['order_date']).dt.strftime('%d-%m')))
    #else:
    #    label = ['Week 1', 'Week 2', 'Week 3']
    
    #ax.set_xticks(range(len(label)))
    ax.set_xticklabels([])
    ax.set_xlabel(f'Orders Trend of {time}')

    return plt

def best_product(record, time):
    plt.figure(figsize=(7, 7))
    plt.pie(
        record['Total orders'].values,
        labels=record['item_name'].values,
        autopct='%1.1f%%',
        startangle=140
    )
    plt.axis('equal')

    ax = plt.gca()
    ax.set_xlabel(f'Favorite Products of {time}')

    return plt
