import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import pickle
import math
# sys.path.insert(1, r'C:\Users\pyy\Dropbox (CEDIA)\CPR\Model')
from CPR import main
import streamlit as st
import plotly.graph_objects as go

# DEFINE FUNCTIONS

def ask_hh():
    st.markdown("# Respondent")
    d_hh = info_spouse()
    st.markdown("## Do you have a spouse?")
    spouse_ques = st.radio("Select", ["Yes", "No"], index=1)
    d_hh['couple'] =  (spouse_ques == "Yes")
    if d_hh['couple']:
        st.markdown("# Spouse")
        d_hh.update(info_spouse('second'))

    fin_accs = ["rrsp", "tfsa", "other_reg", "unreg"]
    fin_prods = ["crsa", "hipsa", "mf", "stocks", "bonds", "gic", "cvplp", "isf", "etf"]
    fin_list = []
    for i in fin_accs:
        for j in fin_prods:
            fin_list.append(i+"_"+j)
            if d_hh['couple']:
                fin_list.append("s_"+i+"_"+j)

    fin_dict = {}
    for i in fin_list:
        if i in d_hh:
            fin_dict[i] = d_hh[i]
        else:
            fin_dict[i] = 0
    prod_dict = pd.DataFrame([fin_dict])
    prod_dict.columns = [i.split("_")[-1] for i in list(prod_dict)]
    prod_dict = prod_dict.groupby(axis=1, level=0).sum()
    prod_dict = prod_dict.transpose().to_dict()[0]

    st.markdown("# Household")
    d_hh.update(info_hh(prod_dict))
    d_hh['weight'] = 1
    return d_hh

def info_spouse(which='first', step_amount=100):
    d = {}
    d['byear'] = st.number_input("Birth Year", min_value=1900, max_value=2020,
                                 key="byear_"+which, value=1980)
    if d['byear'] < 1957:
        st.error("Sorry, the simulator only works for people born after 1956")
        st.stop()
        
    d_gender = {'female': 'Female', 'male': 'Male'}
    d['sex'] = st.radio("Gender", options=list(d_gender.keys()),
                        format_func=lambda x: d_gender[x], key="sex_"+which, index=1)
    age = 2021 - d['byear']
    d['ret_age'] = st.number_input("Retirement Age", min_value=age+1, key="ret_age_"+which, value=max(age + 1, 65))
    d['claim_age_cpp'] = min(d['ret_age'], 70)
    st.success("claim age cpp: {} ({})&nbsp;&nbsp;&nbsp;&nbsp; claim age OAS: 65".format(d["claim_age_cpp"], message_cpp(d["ret_age"])))
    d_education = {'No certificate, diploma or degree': 'less than high school',
                    'Secondary (high) school diploma or equivalency certificate': 'high school',
                    'Certificate of Apprenticeship or Certificate of Qualification': 'post-secondary',
                    'Trades certificate or diploma other than Certificate of Apprenticeship or Certificate of Qualification)': 'post-secondary',
                    'Program of 3 months to less than 1 year (College, CEGEP and other non-university certificates or diplomas)': 'post-secondary',
                    'Program of 1 to 2 years (College, CEGEP and other non-university certificates or diplomas)': 'post-secondary',
                    'Program of more than 2 years (College, CEGEP and other non-university certificates or diplomas)': 'post-secondary',
                    'University certificate or diploma below bachelor level': 'university',
                    "Bachelor's degree": 'university',
                    "Master's degree": 'university',
                    'University certificate or diploma above bachelor level': 'university',
                    'Degree in medicine, dentistry, veterinary medicine or optometry': 'university',
                    'Earned doctorate': 'university'}
    degree = st.selectbox("Education (highest degree obtained)", list(d_education.keys()), key="education_"+which)
    d['education'] = d_education[degree]
    d['init_wage'] = st.number_input("Annual Earnings for 2020", min_value=0, step=step_amount, key="init_wage_"+which, value=50000)

    pension = st.radio("Do you currently (in 2020) receive a pension?", ["Yes", "No"], key="pension_radio_"+which, index=1)
    if pension == "Yes":
        d['pension'] = st.number_input("Yearly amount of pension",  min_value=0, step=step_amount, key="pension_"+which, value=0)   

    savings_plan = st.radio("Do you have any savings or plan to save in the future?", ["Yes", "No"], 
                            key="savings_plan_"+which, index=1)
    if savings_plan == "Yes":
        d.update(fin_accounts(which=which))
    else:
        d_fin_details =  {key: 0 for key in ['cap_gains_unreg', 'realized_losses_unreg', 'init_room_rrsp', 'init_room_tfsa']}
        d.update(d_fin_details)

    db_pension = st.radio("Will you receive a DB pension from your current or previous employer", ["Yes", "No"], 
                            key="db_pension_"+which, index=1)
    if db_pension == "Yes":
        st.markdown("### DB Pension")
        d['replacement_rate_db'] = st.slider(
            "Replacement rate of current DB (in %)", min_value=0.0, max_value=70.0,
            step=0.5, key="replacement_rate_db_" + which, value=0.0) / 100
        d['rate_employee_db'] = st.slider(
            "Contribution rate employee of current DB (in %)", min_value=0.0,
            max_value=9.0, step=0.5, key="rate_employee_db_"+which, value=5.0) / 100
        d['income_previous_db'] = st.number_input(
            "Amount of DB pension from previous employer", min_value=0,
            step=step_amount, key="income_previous_db_" + which)

    dc_pension = st.radio("Do you have a DC pension from current or previous employer",
                          ["Yes", "No"], key="dc_pension_"+which, index=1)
    if dc_pension == "Yes":
        st.markdown("### DC Pension")
        d['init_dc'] = st.number_input("Current balance", min_value=0,
                                       step=step_amount, value=0, key="init_dc_" + which)
        d['rate_employee_dc'] = st.slider(
            "Contribution rate employee of current DC (in %)", min_value=0.0,
            max_value=30.0, step=0.5, key="rate_employee_dc_"+which, value=5.0) / 100
        d['rate_employer_dc'] = st.slider(
            "Contribution rate  employer of current DC (in %)", min_value=0.0,
            max_value=30.0, step=0.5, key="rate_employer_dc_"+which, value=5.0) / 100
        if d['rate_employee_dc'] + d['rate_employer_dc'] > 0.18:
            st.warning("**Warning:** Tax legislation caps the combined employee-employer contribution rate at 18% of earnings")
        
    if which == 'second':
        d = {'s_' + k: v for k, v in d.items()}
    
    return d

def info_hh(prod_dict, step_amount=100):
    d_others = {}
    st.markdown("### Which province do you live in?")
    d_prov = {"qc": "QC", "on": "Other (using the ON tax system)"}
    d_others['prov'] = st.selectbox("Province", options=list(d_prov.keys()),
                                    format_func=lambda x: d_prov[x], key="prov")
    d_others.update(mix_fee(prod_dict))
    st.markdown("### Residences")
    for which in ['first', 'second']:
        which_str = "Do you own a "+which+" residence?"
        res = st.radio(which_str, ["Yes", "No"], key=which, index=1)
        if res == "Yes":
            d_others.update(info_residence(which))

    st.markdown("### Business")
    business = st.radio("Do you own a business?", ["Yes", "No"], key="business", index=1)
    if business == "Yes":
        d_others['business'] = st.number_input("Value of the business in 2020",
                                               min_value=0, step=step_amount,
                                               key="business_value")
        
        sell_business = st.radio("Do you plan to sell your business upon retirement?",
                 ["Yes", "No"], key="business", index=1)
        if sell_business == 'Yes':
            user_options['sell_business'] = True
            d_others['price_business'] = st.number_input(
                "Buying price of the business", min_value=0, step=step_amount,
                key="business_price")

    st.markdown("### Debts other than mortgage")
    mortgage = st.radio("Do you have any debt other than mortgage?",
                        ["Yes", "No"], key="mortgage", index=1)
    if mortgage == "Yes":
        d_others.update(debts())
    return d_others

def message_cpp(ret_age):
    message = 'Equal to retirement age'
    if ret_age > 70:
        message += ', but at most 70'
    elif ret_age < 60:
        message += ', but at least 60'
    return message

def debts(step_amount=100):
    d_debts = {}
    dbt_dict = {'credit card debt':'credit_card', 'a personal loan':'personal_loan', 
                'a student loan':'student_loan', 'a car loan':'car_loan',
                'a credit line':'credit_line', 'other debt':'other_debt'}
    l_debts = ['credit_card', 'personal_loan', 'student_loan', 'car_loan',
               'credit_line', 'other_debt']
    l_names = ['credit card debt', 'a personal loan', 'a student loan',
               'a car loan', 'a credit line', 'other debt']

    debt_names = list(dbt_dict.keys()) #addition
    debt_list = st.multiselect(label="Select your debts", options=debt_names,
                               key="debt_names") #addition
    l_names = debt_list #addition

    for i in l_names:
        st.markdown("### {}".format(i))
        d_debts[dbt_dict[i]] = st.number_input("Balance", min_value=0, step=step_amount, key="debt_"+dbt_dict[i])
        d_debts[dbt_dict[i]+"_payment"] = st.number_input("Monthly payment", min_value=0, step=step_amount, 
                                                            key="debt_payment_"+dbt_dict[i], max_value=d_debts[dbt_dict[i]])

    for key in [dbt_dict[i] for i in l_names]: #addition
        if d_debts[key] == 0:
            d_debts.pop(key, None)
            d_debts.pop(key+"_payment", None)
    return d_debts

def info_residence(which, step_amount=1000):
    d_res = {}
    res = "the " + which + " residence"
    sell = st.radio("Do you plan to sell it upon retirement?", ["Yes", "No"],
                    key=which+"_sell", index=1)
    if sell == "Yes":
        user_options[f'sell_{which}_resid'] = True
        res_value_str = "Value"
        d_res[f'{which}_residence'] = st.number_input(res_value_str, min_value=0,
                                                      step=step_amount, key="res_value_"+which)
    else:
        d_res[f'{which}_residence'] = 0
    res_buy_str = "Buying price"
    if which == 'first':
        d_res[f'price_{which}_residence'] = d_res[f'{which}_residence']  # doesn't matter since cap gain not taxed
    else:
        if sell == "Yes":
            d_res[f'price_{which}_residence'] = st.number_input(
                res_buy_str, min_value=0, step=step_amount, key="res_buy_"+which)
        else:
            d_res[f'price_{which}_residence'] = 0

    res_mortgage_str = "Outstanding mortgage"
    d_res[f'{which}_mortgage'] = st.number_input(res_mortgage_str, min_value=0, step=step_amount, key="res_mortgage_"+which)
    res_mortgage_payment_str = "Monthly payment on mortgage"
    d_res[f'{which}_mortgage_payment'] = st.number_input(res_mortgage_payment_str, min_value=0, step=step_amount, 
                                                        key="res_mortgage_payment_"+which)
    return d_res

def mix_fee(prod_dict):
    df = pd.read_csv('mix_fee_assets.csv', index_col=0, usecols=range(0, 5))
    d_investments = {}
    total_sum = sum(prod_dict.values())
    # portfolio for people without current savings (loosely calibrated from PowerCorp's database)
    if total_sum == 0:
        d_mix_fee = {}
        d_mix_fee['fee_equity'] = 0.005
        d_mix_fee["mix_bills"] = 0.60
        d_mix_fee["mix_bonds"] = 0.15
        d_mix_fee["mix_equity"] = 0.25
        d_mix_fee["fee"] = 0.015
        
    else:
        d_investments["Checking or regular savings account"] = prod_dict["crsa"]/total_sum
        d_investments["High interest/premium savings account"] = prod_dict["hipsa"]/total_sum
        d_investments["Mutual funds"] = prod_dict["mf"]/total_sum
        d_investments["Stocks"] = prod_dict["stocks"]/total_sum
        d_investments["Bonds"] = prod_dict["bonds"]/total_sum
        d_investments["GICs"] = prod_dict["gic"]/total_sum
        d_investments["Cash value of permanent life policy"] = prod_dict["cvplp"]/total_sum
        d_investments["Individual segregated funds"] = prod_dict["isf"]/total_sum
        d_investments["ETFs"] = prod_dict["etf"]/total_sum
        d_mix_fee = {key: 0 for key in df.columns}
        df['fraction'] = pd.Series(d_investments)
        for key in d_mix_fee:
            d_mix_fee[key] = (df[key] * df.fraction).sum()
        if (df.equity * df.fraction).sum() == 0:
            d_mix_fee['fee_equity'] = 0
        else:
            d_mix_fee['fee_equity'] = (df.equity * df.fraction * df.fee).sum() / (df.equity * df.fraction).sum()
        if math.isnan(d_mix_fee['fee_equity']):
            d_mix_fee['fee_equity'] = 0
        d_mix_fee["mix_bills"] = d_mix_fee.pop("bills")
        d_mix_fee["mix_bonds"] = d_mix_fee.pop("bonds")
        d_mix_fee["mix_equity"] = d_mix_fee.pop("equity")
        
    return d_mix_fee

def fin_accounts(which, step_amount=100):
    d_fin = {}
    st.markdown("### Savings account")
    saving_plan_select = st.multiselect(label="Select your savings accounts", 
                                        options=["RRSP", "TFSA", "Other Reg", "Unreg"], key="fin_acc_"+which) #addition
    accs = saving_plan_select #addition
    #accs = ["rrsp", "tfsa", "other_reg", "unreg"]
    acc_cap = {"rrsp": "RRSP", "tfsa": "TFSA", "other_reg": "Other Reg", "unreg": "Unreg"}
    acc_cap_rev = {v: k for k, v in acc_cap.items()} #addition
    accs = [acc_cap_rev[i] for i in accs] #addition
    for i in accs:
        st.markdown("### {}".format(acc_cap[i]))
        d_fin["bal_"+i] = st.number_input(
            "Amount in {} account".format(acc_cap[i]), value=0, min_value=0, step=step_amount,
            key="bal_"+i+"_"+which)
        d_fin["cont_rate_"+i] = st.number_input(
            "Fraction of your earnings you plan to save in your {} account (in %)".format(
                acc_cap[i]), value=0, min_value=0, max_value=100, step=1, key="cont_rate_"+i+"_"+which)
        d_fin["cont_rate_"+i] /= 100.0
        d_fin["withdrawal_"+i] = st.number_input(
            "Amount of your {} account you plan to spend (in $)".format(acc_cap[i]),
            value=0, min_value=0, step=step_amount, key="withdrawal_"+i+"_"+which)
        if i in ["rrsp", "tfsa"]:
            d_fin["init_room_"+i] = st.number_input(
                "Contribution room for {} in 2020".format(acc_cap[i]),
                value=0, min_value=0, step=step_amount, key="init_room_"+i+"_"+which)

        if d_fin["bal_"+i] > 0:
            d_fin.update(financial_products(i, d_fin["bal_"+i], which, step_amount=step_amount))

    st.markdown("### Gains and Losses in Unregistered Account")
    d_fin['cap_gains_unreg'] = st.number_input(
        "Balance of unrealized capital gains as of December 31, 2020",
        value=0, min_value=0, step=step_amount, key="cap_gains_unreg_"+which)
    d_fin['realized_losses_unreg'] = st.number_input(
        "Realized losses in capital on unregistered account as of December 31, 2020",
        value=0, min_value=0, step=step_amount, key="realized_losses_unreg_"+which)
    return d_fin

def financial_products(account, balance, which, step_amount=100):
    d_fp = {}
    total_fp = 0
    acc_cap = {"rrsp": "RRSP", "tfsa": "TFSA", "other_reg": "Other Reg", "unreg": "Unreg"}
    st.markdown("### {} - Financial Products".format(acc_cap[account]))
    fin_prods = ["crsa", "hipsa", "mf", "stocks", "bonds", "gic", "cvplp", "isf", "etf"]
    fin_prods_dict = {"crsa": "Amount in Checking or regular savings account",
                      "hipsa": "Amount in High interest/premium savings account",
                      "mf": "Amount in Mutual funds",
                      "stocks": "Amount in Stocks",
                      "bonds": "Amount in Bonds",
                      "gic": "Amount in GICs",
                      "etf": "Amount in ETFs"}

    fin_prods_rev = {v: k for k, v in fin_prods_dict.items()} #addition
    fin_prod_list = list(fin_prods_rev.keys()) #addition
    fin_prod_select = st.multiselect(label="Select financial products", 
                                    options=fin_prod_list, key="fin_prod_list_"+account+"_"+which) #addition
    fin_prods = [fin_prods_rev[i] for i in fin_prod_select] #addition
    for i in fin_prods:
        d_fp[account+"_"+i] = st.number_input(fin_prods_dict[i], value=0, min_value=0,
                                              step=step_amount, key=account+"_"+i+"_"+which)
        total_fp += d_fp[account+"_"+i]

    if total_fp != balance:
        st.error("Total amount in financial products ({} $) is not equal to amount in financial account ({} $)".format(
                total_fp, balance))
        st.stop()
    return d_fp

def create_dataframe(d_hh):
    l_p = ['byear', 'sex', 'ret_age', 'education', 'init_wage', 'pension', 'bal_rrsp', 'bal_tfsa', 'bal_other_reg', 'bal_unreg',
            'cont_rate_rrsp', 'cont_rate_tfsa', 'cont_rate_other_reg', 'cont_rate_unreg', 'withdrawal_rrsp', 'withdrawal_tfsa',
            'withdrawal_other_reg', 'withdrawal_unreg', 'replacement_rate_db',
            'rate_employee_db', 'income_previous_db', 'init_dc', 'rate_employee_dc', 'rate_employer_dc', 'claim_age_cpp',
            'cap_gains_unreg', 'realized_losses_unreg', 'init_room_rrsp', 'init_room_tfsa']
    l_sp = ['s_' + var for var in l_p]
    l_hh = ['weight', 'couple', 'prov', 'first_residence', 'second_residence', 'price_first_residence', 'price_second_residence', 
            'business', 'price_business', 'mix_bonds', 'mix_bills', 'mix_equity', 'fee', 'fee_equity', 'credit_card', 
            'personal_loan', 'student_loan', 'car_loan', 'credit_line', 'first_mortgage', 'second_mortgage', 'other_debt', 
            'credit_card_payment', 'personal_loan_payment', 'student_loan_payment', 'car_loan_payment', 'credit_line_payment',
            'first_mortgage_payment', 'second_mortgage_payment', 'other_debt_payment']   
    
    return pd.DataFrame(d_hh, columns=l_p + l_sp + l_hh, index=[0])

def prepare_RRI(df):
    results = main.run_simulations(df, nsim=20, n_jobs=1, non_stochastic=False,
                                   base_year=2020, **user_options)
    results.merge()
    df_res = results.df_merged
    cons_floor = 0
    if len(df_res[df_res["cons_bef"] < cons_floor]):
        st.error("Consumption before retirement is negative: savings or debt payments are too high")
        st.stop()
    if len(df_res[df_res["cons_after"] < cons_floor]):
        st.error("Consumption after retirement is negative: debt payments are too high")
        st.stop()
    df_res['RRI'] = df_res.cons_after / df_res.cons_bef * 100
    return df_res

def check_cons_positive(df, cons_floor = 0):
    if len(df[df["cons_bef"] < cons_floor]):
        st.error("Consumption before retirement is negative: savings or debt payments are too high")
        st.stop()
    if len(df[df["cons_after"] < cons_floor]):
        st.error("Consumption after retirement is negative: debt payments are too high")
        st.stop()

def create_data_changes(df):
    df_change = pd.DataFrame(np.repeat(df.values, 5, axis=0), columns=df.columns)
    df_change.cont_rate_rrsp += np.array([0, 0.05, 0.10, 0, 0])
    df_change.ret_age += np.array([0, 0, 0, -2, 2])
    if any(df_change.couple) is True:
        df_change.s_ret_age += np.array([0, 0, 0, -2, 2])
    return df_change


# GRAPHS

def show_plot_button(df):
    
    # stochastic results
    nsim = 25
    results = main.run_simulations(df, nsim=nsim, n_jobs=1, non_stochastic=False,
                                   base_year=2020, **user_options)
    df_output = results.output
    check_cons_positive(df_output, cons_floor = 0)
    df_output['RRI'] = (df_output.cons_after / df_output.cons_bef * 100).round(1)

    # figure stochastic
    fig = go.Figure()
    fig.add_scatter(x=df_output.cons_bef, y=df_output.cons_after,
                    mode='markers', 
                    marker=dict(size=10, line=dict(color='MediumPurple', width=2)),
                    hovertemplate=
                    'cons. bef. ret: $%{x:,.0f} <br>'+
                    'cons. after ret: $%{y:,.0f} <br>'+
                    '%{text}</b>' + 
                    '<extra></extra>',
                    text=['RRI: {:.0f}'.format(x) for x in df_output.RRI.values],
                    showlegend = False)

    fig.add_scatter(x=[df_output.cons_bef.mean()], y=[df_output.cons_after.mean()],
                    mode='markers', name='mean consumptions',
                    marker_size=15, marker_symbol='x',
                    hovertemplate=
                    'cons. bef. ret: $%{x:,.0f} <br>'+
                    'cons. after ret: $%{y:,.0f} <br>'+
                    '%{text}</b>' + 
                    '<extra></extra>',
                    text=['RRI: {:.0f}'.format(x) for x in df_output.RRI.values])

    cons_bef = np.array([df_output.cons_bef.min(), df_output.cons_bef.max()])
    fig.add_trace(go.Scatter(x=cons_bef, y=.65 * cons_bef,
                            mode='lines', name="RRI = 65",
                            line=dict(color="RoyalBlue", width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=cons_bef, y=.80 * cons_bef,
                            mode='lines', name="RRI = 80",
                            line=dict(color="Green", width=2, dash='dot')))
    fig.update_layout(height=500, width=700,
                    title={'text': f"Consumptions ({nsim} realizations)",
                            'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
                    xaxis_title="before retirement",
                    yaxis_title="after retirement",
                    font=dict(family="Courier New, monospace",
                                size=14, color="RebeccaPurple"),
                    legend={'traceorder':'reversed'})
    st.plotly_chart(fig)
    
    ## to add text: st.write('whatever i want TBC')

    # create data with changes in contribution rate rrsp and retirement age
    df_change = create_data_changes(df)
    results = main.run_simulations(df_change, nsim=1, n_jobs=1,non_stochastic=True,
                                   base_year=2020, **user_options)
    results.merge()
    df_change = results.df_merged
    
    
    # graph changes in contribution rate rrsp and retirement age
    names = ['initial case', 'contrib rrsp + 5%', 'contrib rrsp + 10%',
             'ret age - 2', 'ret age + 2']

    fig = go.Figure()

    l_cons_bef = []
    colors = ['black', 'red', 'green', 'red', 'green']
    symbols = ['circle', 'diamond', 'diamond', 'x', 'x']
    for index, row in df_change.iterrows():
        l_cons_bef.append(row['cons_bef'])
        rri = row['cons_after'] / row['cons_bef'] * 100
        fig.add_scatter(x=[row['cons_bef']], y=[row['cons_after']],
                        mode='markers',
                        marker_color=colors[index],
                        marker_symbol = symbols[index],
                        marker=dict(size=12, line=dict(color='MediumPurple', width=2)),
                        name=names[index],
                        hovertemplate=
                        '%{text} <br>'
                        'cons. bef. ret: $%{x:,.0f} <br>'+
                        'cons. after ret: $%{y:,.0f} <br>'+
                        '<extra></extra>',
                        text = [f'<b>{names[index]}</b> <br />RRI: {rri:.0f}'],
                        showlegend = True)

    cons_bef = np.array([min(l_cons_bef), max(l_cons_bef)])
    fig.add_trace(go.Scatter(x=cons_bef, y=.80 * cons_bef,
                            mode='lines', name="RRI = 80",
                            line=dict(color="Green", width=1, dash='dot')))

    fig.add_trace(go.Scatter(x=cons_bef, y=.65 * cons_bef,
                            mode='lines', name="RRI = 65",
                            line=dict(color="RoyalBlue", width=1, dash='dash')))


    fig.update_layout(height=500, width=700,
                    title={'text': f"Consumptions",
                            'x': 0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'},
                    xaxis_title="before retirement",
                    yaxis_title="after retirement",
                    font=dict(family="Courier New, monospace",
                                size=14,
                                color="RebeccaPurple"))
    st.plotly_chart(fig)

    
    # Income decomposition
    # prepare data
    hhold = df_change.loc[0, :]
    pension = hhold['pension_after']
    annuity = hhold['annuity_rrsp_after'] + hhold['annuity_rpp_dc_after'] + hhold['annuity_non_rrsp_after']
    consumption = hhold['cons_after']
    debt_payments = hhold['debt_payments_after']
    imputed_rent = hhold['imputed_rent_after']
    net_liabilities = hhold['fam_net_tax_liability_after']
    cpp = hhold['cpp_after']
    gis = hhold['gis_after']
    oas = hhold['oas_after']
    rpp_db = hhold['rpp_db_benefits_after']
    business_dividends = hhold['business_dividends_after']
    
    if hhold['couple']:
        pension += hhold['s_pension_after']
        annuity += hhold['s_annuity_rrsp_after'] + hhold['s_annuity_rpp_dc_after'] + hhold['s_annuity_non_rrsp_after']
        cpp += hhold['s_cpp_after']
        gis += hhold['s_gis_after']
        oas += hhold['s_oas_after']
        rpp_db += hhold['s_rpp_db_benefits_after']
        business_dividends += hhold['s_business_dividends_after']
    income = oas + gis + cpp + rpp_db + annuity + pension

    label = ['Income', # 0
            'OAS', 'GIS', 'CPP', 'RPP DB', 'Annuity', 'Pension', 'Business Dividends', # 1 to 7
            'Consumption', 'Imputed Rent', 'Debt payments', # 8 - 10
            'Net tax liability']  # 11 could also enter income (invert source and target)

    if net_liabilities > 0:
        source = [1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0]
        target = [0, 0, 0, 0, 0, 0, 0, 8, 9, 10, 11]
        value =  [oas, gis, cpp, rpp_db, annuity, pension, business_dividends,
                  consumption, imputed_rent, debt_payments, net_liabilities]
    else:
        source = [1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 11]
        target = [0, 0, 0, 0, 0, 0, 0, 8, 9, 10, 0]
        value =  [oas, gis, cpp, rpp_db, annuity, pension, business_dividends,
                  consumption, imputed_rent, debt_payments, -net_liabilities]

    # data to dict, dict to sankey
    link = dict(source = source,
                target = target,
                value = value,
                hovertemplate='$%{value:,.0f}<extra></extra>')
    node = dict(label = label, pad=20, thickness=50,
                hovertemplate='$%{value:,.0f}<extra></extra>')

    data = go.Sankey(link=link, node=node)
    # plot
    fig = go.Figure(data)
    fig.update_layout(height=450, width=750,
                    title_text="Retirement income decomposition",
                    font_size=16)
    st.plotly_chart(fig)
    
    
# SCRIPT INTERFACE

user_options = {'sell_business': False,
                'sell_first_resid': False,
                'sell_second_resid': False}

st.markdown(f"""<style>
    .reportview-container .main .block-container{{
    max-width: 1280px;
    padding-top: 0em;
    }}</style>""", unsafe_allow_html=True)

st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>CPR Simulator</h1>", unsafe_allow_html=True)

st.sidebar.markdown("# DISCLAIMER")
st.sidebar.markdown("This tool uses the freely available [Canadians’ Preparation for Retirement (CPR) calculator](https://ire.hec.ca/en/canadians-preparation-retirement-cpr), \
    developed by a team at [HEC Montréal](https://www.hec.ca/en/)’s Retirement and Savings Institute with financial support from the \
    [Global Risk Institute](https://globalriskinstitute.org/)’s [National Pension Hub](https://globalriskinstitute.org/national-pension-hub/).")
st.sidebar.markdown("It is provided “as is” for personal use only, without any warranty regarding its accuracy, appropriateness, completeness or any other quality.")
st.sidebar.markdown("Its results are deemed to be general information on retirement preparation and should not be construed as financial advice; qualified financial advice should be sought before making any financial decision based on this tool.")
st.sidebar.markdown("Use of the tool implies the acceptance of the foregoing terms and constitutes an acknowledgement that this disclaimer has been read and understood.")

col_p1, _, col_p2 = st.beta_columns([0.465, 0.025, 0.51])

with col_p1:
    d_hh = ask_hh()
    df = create_dataframe(d_hh)
    df = df.fillna(0)
    
    # set all NaNs to zero
    fin_acc_cols = ['bal_rrsp', 'bal_tfsa', 'bal_other_reg', 'bal_unreg', 
                    'cont_rate_rrsp', 'cont_rate_tfsa', 'cont_rate_other_reg',
                    'cont_rate_unreg', 'withdrawal_rrsp', 'withdrawal_tfsa', 
                    'withdrawal_other_reg', 'withdrawal_unreg', 'cap_gains_unreg',
                    'realized_losses_unreg', 'init_room_rrsp', 'init_room_tfsa']
    
    if df['couple'][0]:
        s_fin_acc_cols = ['s_' + col for col in fin_acc_cols]
        fin_acc_cols += s_fin_acc_cols
    df[fin_acc_cols] = df[fin_acc_cols].fillna(0)

with col_p2:
    if st.button("Update visualizations", False):
        show_plot_button(df)

if st.button("Show visualizations", False):
    with col_p2:
        show_plot_button(df)