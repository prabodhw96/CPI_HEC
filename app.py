import numpy as np
import pandas as pd
import tkinter
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import pickle
from CPR import main
import streamlit as st

def ask_hh():
    st.markdown("# Respondent")
    d_hh = info_spouse()
    st.markdown("## Do you have a spouse?")
    spouse_ques = st.radio("Select", ["Yes", "No"], index=0)
    spouse_dict = {}
    spouse_dict["Yes"] = True
    spouse_dict["No"] = False
    d_hh['couple'] =  spouse_dict[spouse_ques]
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
        if i in list(d_hh.keys()):
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

def message_cpp(ret_age):
    message = 'Equal to retirement age'
    if ret_age > 70:
        message += ', but at most 70'
    elif ret_age < 60:
        message += ', but at least 60'
    return message

def info_spouse(which='first'):
    d = {}
    d['byear'] = st.number_input("Birth Year", min_value=1900, max_value=2020, key="byear_"+which, value=1980)
    d_gender = {'female': 'Female', 'male': 'Male'}
    d['sex'] = st.radio("Gender", options=list(d_gender.keys()), format_func=lambda x: d_gender[x], key="sex_"+which, index=1)
    d['ret_age'] = st.number_input("Retirement Age", min_value=2021-d['byear']+1, key="ret_age_"+which, value=65)
    d['claim_age_cpp'] = min(d['ret_age'], 70)
    st.success("claim age cpp: {} ({})".format(d["claim_age_cpp"], message_cpp(d["ret_age"])))
    st.success("claim age OAS: 65")
    d['education'] = st.selectbox("Education (highest degree obtained)", 
        ["university", "post-secondary", "high school", "less than high school"], key="education_"+which)
    d['init_wage'] = st.number_input("Annual Earnings for 2018", min_value=0, step=500, key="init_wage_"+which, value=50000)

    pension = st.radio("Do you currently (in 2020) receive a pension?", ["Yes", "No"], key="pension_radio_"+which, index=1)
    if pension == "Yes":
        d['pension'] = st.number_input("Yearly amount of pension",  min_value=0, step=500, key="pension_"+which, value=0)

    savings_plan = st.radio("Do you have any savings or plan to save in the future?", ["Yes", "No"], key="savings_plan_"+which, index=1)
    if savings_plan == "Yes":
        d.update(fin_accounts(which=which))
    else:
        d_fin_details =  {key: 0 for key in ['cap_gains_unreg', 'realized_losses_unreg', 'init_room_rrsp', 'init_room_tfsa']}
        d.update(d_fin_details)

    db_pension = st.radio("Will you receive a DB pension from your current or previous employer", ["Yes", "No"], key="db_pension_"+which, index=1)
    if db_pension == "Yes":
        st.markdown("### DB Pension")
        d['replacement_rate_db'] = st.slider("Replacement rate of current DB (in %)",
                                                 min_value=0, max_value=70, step=1,
                                                 key="replacement_rate_db_" + which, value=0)
        d['replacement_rate_db'] /= 100
        d['rate_employee_db'] = st.slider("Contribution rate employee of current DB (in %)",
                                              min_value=0.00, max_value=9.00, step=0.5,
                                              key="rate_employee_db_"+which, value=5.00)
        d['rate_employee_db'] /= 100
        d['income_previous_db'] = st.number_input("Amount of DB pension from previous employer",
                                                      min_value=0.0, step=500.0, key="income_previous_db_"+which)

    dc_pension = st.radio("Do you have a DC pension from current or previous employer",
                          ["Yes", "No"], key="dc_pension_"+which, index=1)
    if dc_pension == "Yes":
        st.markdown("### DC Pension")
        d['init_dc'] = st.number_input("Annual amount of current pension", min_value=0,
                                           step=500, value=0, key="init_dc_" + which)
        d['rate_employee_dc'] = st.slider("Contribution rate employee of current DC (in %)",
                                              min_value=0, max_value=30, step=1,
                                              key="rate_employee_dc_"+which, value=5)
        d['rate_employee_dc'] /= 100
        d['rate_employer_dc'] = st.slider("Contribution rate  employer of current DC (in %)",
                                              min_value=0, max_value=30, step=1,
                                              key="rate_employer_dc_"+which, value=5)
        d['rate_employer_dc'] /= 100
        if d['rate_employee_dc'] + d['rate_employer_dc'] > 0.18:
            st.warning("**Warning:** Tax legislation caps the combined employee-employer contribution rate at 18% of earnings")
        

    if which == 'second':
        d = {'s_' + k: v for k, v in d.items()}
    return d

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

# this function should check if the keys are right!
def change_info(d_hh, **kwargs):
    return d_hh.update(kwargs)

def debts():
    d_debts = {}
    dbt_dict = {'credit card debt':'credit_card', 'a personal loan':'personal_loan', 'a student loan':'student_loan',
                'a car loan':'car_loan', 'a credit line':'credit_line', 'other debt':'other_debt'}
    l_debts = ['credit_card', 'personal_loan', 'student_loan', 'car_loan', 'credit_line', 'other_debt']
    l_names = ['credit card debt', 'a personal loan', 'a student loan', 'a car loan', 'a credit line', 'other debt']

    st.markdown("### Credit card debt")
    d_debts["credit_card"] = st.number_input("Balance", min_value=0.0, step=500.0, key="debt_credit_card")
    d_debts["credit_card_payment"] = st.number_input("Monthly payment", min_value=0.0, step=100.0, key="debt_payment_credit_card", max_value=d_debts["credit_card"])

    st.markdown("### Personal Loan")
    d_debts["personal_loan"] = st.number_input("Balance", min_value=0.0, step=500.0, key="debt_personal_loan")
    d_debts["personal_loan_payment"] = st.number_input("Monthly payment", min_value=0.0, step=100.0, key="debt_payment_personal_loan", max_value=d_debts["personal_loan"])

    st.markdown("### Student Loan")
    d_debts["student_loan"] = st.number_input("Balance", min_value=0.0, step=500.0, key="debt_student_loan")
    d_debts["student_loan_payment"] = st.number_input("Monthly payment", min_value=0.0, step=100.0, key="debt_payment_student_loan", max_value=d_debts["student_loan"])

    st.markdown("### Car Loan")
    d_debts["car_loan"] = st.number_input("Balance", min_value=0.0, step=500.0, key="debt_car_loan")
    d_debts["car_loan_payment"] = st.number_input("Monthly payment", min_value=0.0, step=100.0, key="debt_payment_car_loan", max_value=d_debts["car_loan"])

    st.markdown("### Credit Line")
    d_debts["credit_line"] = st.number_input("Balance", min_value=0.0, step=500.0, key="debt_credit_line")
    d_debts["credit_line_payment"] = st.number_input("Monthly payment", min_value=0.0, step=100.0, key="debt_payment_credit_line", max_value=d_debts["credit_line"])

    st.markdown("### Other debt")
    d_debts["other_debt"] = st.number_input("Balance", min_value=0.0, step=500.0, key="debt_other_debt")
    d_debts["other_debt_payment"] = st.number_input("Monthly payment", min_value=0.0, step=100.0, key="debt_payment_other_debt", max_value=d_debts["other_debt"])

    for key in ["credit_card", "personal_loan", "student_loan", "car_loan", "credit_line", "other_debt"]:
        if d_debts[key] == 0:
            d_debts.pop(key, None)
            d_debts.pop(key+"_payment", None)

    return d_debts

def info_residence(which):
    d_res = {}
    res = "the "+which+" residence"
    res_value_str = "Value"
    d_res[f'{which}_residence'] = st.number_input(res_value_str, min_value=0.0, step=500.0, key="res_value_"+which)
    res_buy_str = "Buying price"
    d_res[f'price_{which}_residence'] = st.number_input(res_buy_str, min_value=0.0, step=500.0, key="res_buy_"+which)
    res_mortgage_str = "Outstanding mortgage"
    d_res[f'{which}_mortgage'] = st.number_input(res_mortgage_str, min_value=0.0, step=500.0, key="res_mortgage_"+which)
    res_mortgage_payment_str = "Monthly payment on mortgage"
    d_res[f'{which}_mortgage_payment'] = st.number_input(res_mortgage_payment_str, min_value=0.0, step=500.0, key="res_mortgage_payment_"+which)
    return d_res

def mix_fee():
    st.markdown("### Investments")
    df = pd.read_csv('mix_fee_assets.csv', index_col=0, usecols=range(0, 5))
    d_investments = {}
    st.markdown("#### Checking or regular savings account")
    d_investments["Checking or regular savings account"] = st.slider("Fraction of portfolio invested", min_value=0.0, max_value=100.0, value=10.0, step=0.5, key="mix_fee_checking")
    d_investments["Checking or regular savings account"] /= 100

    st.markdown("#### High interest/premium savings account")
    d_investments["High interest/premium savings account"] = st.slider("Fraction of portfolio invested", min_value=0.0, max_value=100.0, value=10.0, step=0.5, key="mix_fee_premium")
    d_investments["High interest/premium savings account"] /= 100

    st.markdown("#### Mutual funds")
    d_investments["Mutual funds"] = st.slider("Fraction of portfolio invested", min_value=0.0, max_value=100.0, value=10.0, step=0.5, key="mix_fee_mf")
    d_investments["Mutual funds"] /= 100

    st.markdown("#### Stocks")
    d_investments["Stocks"] = st.slider("Fraction of portfolio invested", min_value=0.0, max_value=100.0, value=10.0, step=0.5, key="mix_fee_stocks")
    d_investments["Stocks"] /= 100

    st.markdown("#### Bonds")
    d_investments["Bonds"] = st.slider("Fraction of portfolio invested", min_value=0.0, max_value=100.0, value=10.0, step=0.5, key="mix_fee_bonds")
    d_investments["Bonds"] /= 100

    st.markdown("#### GICs")
    d_investments["GICs"] = st.slider("Fraction of portfolio invested", min_value=0.0, max_value=100.0, value=10.0, step=0.5, key="mix_fee_gic")
    d_investments["GICs"] /= 100

    st.markdown("#### Cash value of permanent life policy")
    d_investments["Cash value of permanent life policy"] = st.slider("Fraction of portfolio invested", min_value=0.0, max_value=100.0, value=10.0, step=0.5, key="mix_fee_policy")
    d_investments["Cash value of permanent life policy"] /= 100

    st.markdown("#### Individual segregated funds")
    d_investments["Individual segregated funds"] = st.slider("Fraction of portfolio invested", min_value=0.0, max_value=100.0, value=10.0, step=0.5, key="mix_fee_segregated")
    d_investments["Individual segregated funds"] /= 100

    st.markdown("#### ETFs")
    d_investments["ETFs"] = st.slider("Fraction of portfolio invested", min_value=0.0, max_value=100.0, value=20.0, step=0.5, key="mix_fee_etf")
    d_investments["ETFs"] /= 100

    d_mix_fee = {key: 0 for key in df.columns}
    df['fraction'] = pd.Series(d_investments)
    if df['fraction'].sum() != 1:
        st.error("Total sum = {}%; does not sum up to 100%".format(round(df['fraction'].sum()*100, 2)))
        st.stop()
    else:
        st.success("Total sum = 100% !")

    for key in d_mix_fee:
        d_mix_fee[key] = (df[key] * df.fraction).sum()
    d_mix_fee['fee_equity'] = (df.equity * df.fraction * df.fee).sum() / (df.equity * df.fraction).sum()
    d_mix_fee['fee_equity'] /= 100.0
    d_mix_fee["mix_bills"] = d_mix_fee.pop("bills")
    d_mix_fee["mix_bills"] /= 100.0
    d_mix_fee["mix_bonds"] = d_mix_fee.pop("bonds")
    d_mix_fee["mix_bonds"] /= 100.0
    d_mix_fee["mix_equity"] = d_mix_fee.pop("equity")
    d_mix_fee["mix_equity"] /= 100.0
    d_mix_fee["fee"] /= 100.0
    return d_mix_fee

def mix_fee(prod_dict):
    df = pd.read_csv('mix_fee_assets.csv', index_col=0, usecols=range(0, 5))
    d_investments = {}
    total_sum = sum(prod_dict.values())
    if total_sum == 0:
        d_mix_fee = {}
        d_mix_fee['fee_equity'] = 0
        d_mix_fee["mix_bills"] = 0
        d_mix_fee["mix_bonds"] = 0
        d_mix_fee["mix_equity"] = 0
        d_mix_fee["fee"] = 0
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
        d_mix_fee['fee_equity'] = (df.equity * df.fraction * df.fee).sum() / (df.equity * df.fraction).sum()
        d_mix_fee['fee_equity'] /= 100.0
        d_mix_fee["mix_bills"] = d_mix_fee.pop("bills")
        d_mix_fee["mix_bills"] /= 100.0
        d_mix_fee["mix_bonds"] = d_mix_fee.pop("bonds")
        d_mix_fee["mix_bonds"] /= 100.0
        d_mix_fee["mix_equity"] = d_mix_fee.pop("equity")
        d_mix_fee["mix_equity"] /= 100.0
        d_mix_fee["fee"] /= 100.0
    return d_mix_fee


def info_hh(prod_dict):
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
        d_others['business'] = st.number_input("Value of the business in 2020", min_value=0.0, step=500.0, key="business_value")
        d_others['price_business'] = st.number_input("Buying price of the business", max_value=0.0, step=500.0, key="business_price")

    mortgage = st.radio("Do you have any debt other than mortgage?", ["Yes", "No"], key="mortgage", index=1)
    if mortgage == "Yes":
        d_others.update(debts())

    return d_others

def fin_accounts(which):
    d_fin = {}
    st.markdown("### Savings account")

    st.markdown("### RRSP")
    d_fin["bal_rrsp"] = st.number_input("Amount in RRSP account", min_value=0.0, step=500.0, key="bal_rrsp_"+which)
    d_fin["cont_rate_rrsp"] = st.number_input("Fraction of your earnings you plan to save in your RRSP account (in %)",
                                                min_value=0.0, max_value=100.0, step=1.0, key="cont_rate_rrsp_"+which)
    d_fin["cont_rate_rrsp"] /= 100
    d_fin["withdrawal_rrsp"] = st.number_input("Amount of your RRSP account you plan to spend (in $)",
                                                   min_value=0.0, step=500.0, key="withdrawal_rrsp_"+which)
    d_fin["init_room_rrsp"] = st.number_input("Contribution room for RRSP in 2020", 
                                                  min_value=0.0, step=500.0, key="init_room_rrsp_"+which)
    total_rrsp = 0
    d_fin["rrsp_crsa"] = st.number_input("Amount in Checking or regular savings account", min_value=0.0, value=0.0,
                                            step=500.0, key="rrsp_crsa_"+which)
    total_rrsp += d_fin["rrsp_crsa"]
    d_fin["rrsp_hipsa"] = st.number_input("Amount in High interest/premium savings account", min_value=0.0, value=0.0,
                                            step=500.0, key="rrsp_hipsa_"+which)
    total_rrsp += d_fin["rrsp_hipsa"]
    d_fin["rrsp_mf"] = st.number_input("Amount in Mutual funds", min_value=0.0, value=0.0, step=500.0, key="rrsp_mf_"+which)
    total_rrsp += d_fin["rrsp_mf"]
    d_fin["rrsp_stocks"] = st.number_input("Amount in Stocks", min_value=0.0, value=0.0, step=500.0, key="rrsp_stocks_"+which)
    total_rrsp += d_fin["rrsp_stocks"]
    d_fin["rrsp_bonds"] = st.number_input("Amount in Bonds", min_value=0.0, value=0.0, step=500.0, key="rrsp_bonds_"+which)
    total_rrsp += d_fin["rrsp_bonds"]
    d_fin["rrsp_gic"] = st.number_input("Amount in GICs", min_value=0.0, value=0.0, step=500.0, key="rrsp_gic_"+which)
    total_rrsp += d_fin["rrsp_gic"]
    d_fin["rrsp_cvplp"] = st.number_input("Amount in Cash value of permanent life policy", min_value=0.0, 
                                            value=0.0, step=500.0, key="rrsp_cvplp_"+which)
    total_rrsp += d_fin["rrsp_cvplp"]
    d_fin["rrsp_isf"] = st.number_input("Amount in Individual segregated funds", min_value=0.0, value=0.0, step=500.0,
                                            key="rrsp_isf_"+which)
    total_rrsp += d_fin["rrsp_isf"]
    d_fin["rrsp_etf"] = st.number_input("Amount in ETFs", min_value=0.0, value=0.0, step=500.0, key="rrsp_etf_"+which)
    total_rrsp += d_fin["rrsp_etf"]
    if total_rrsp != d_fin["bal_rrsp"]:
        st.error("Total amount in financial products ({} $) is not equal to amount in financial account ({} $)".format(total_rrsp, d_fin["bal_rrsp"]))
        st.stop()
    else:
        st.success("Total amount in financial products ({} $) = amount in financial account ({} $)".format(total_rrsp, d_fin["bal_rrsp"]))

    st.markdown("### TFSA")
    d_fin["bal_tfsa"] = st.number_input("Amount in TFSA account", min_value=0.0, step=500.0, key="bal_tfsa_"+which)
    d_fin["cont_rate_tfsa"] = st.number_input("Fraction of your earnings you plan to save in your TFSA account (in %)", min_value=0.00, max_value=100.00, key="cont_rate_tfsa_"+which)
    d_fin["cont_rate_tfsa"] /= 100.0
    d_fin["withdrawal_tfsa"] = st.number_input("Amount of your TFSA account you plan to spend (in $)", min_value=0.0, step=500.0, key="withdrawal_tfsa_"+which)
    d_fin["init_room_tfsa"] = st.number_input("Contribution room for TFSA in 2020",
                                                  min_value=0.0, key="init_room_tfsa_"+which)
    total_tfsa = 0
    d_fin["tfsa_crsa"] = st.number_input("Amount in Checking or regular savings account", min_value=0.0, value=0.0,
                                            step=500.0, key="tfsa_crsa_"+which)
    total_tfsa += d_fin["tfsa_crsa"]
    d_fin["tfsa_hipsa"] = st.number_input("Amount in High interest/premium savings account", min_value=0.0, value=0.0,
                                            step=500.0, key="tfsa_hipsa_"+which)
    total_tfsa += d_fin["tfsa_hipsa"]
    d_fin["tfsa_mf"] = st.number_input("Amount in Mutual funds", min_value=0.0, value=0.0, step=500.0, key="tfsa_mf_"+which)
    total_tfsa += d_fin["tfsa_mf"]
    d_fin["tfsa_stocks"] = st.number_input("Amount in Stocks", min_value=0.0, value=0.0, step=500.0, key="tfsa_stocks_"+which)
    total_tfsa += d_fin["tfsa_stocks"]
    d_fin["tfsa_bonds"] = st.number_input("Amount in Bonds", min_value=0.0, value=0.0, step=500.0, key="tfsa_bonds_"+which)
    total_tfsa += d_fin["tfsa_bonds"]
    d_fin["tfsa_gic"] = st.number_input("Amount in GICs", min_value=0.0, value=0.0, step=500.0, key="tfsa_gic_"+which)
    total_tfsa += d_fin["tfsa_gic"]
    d_fin["tfsa_cvplp"] = st.number_input("Amount in Cash value of permanent life policy", min_value=0.0, 
                                            value=0.0, step=500.0, key="tfsa_cvplp_"+which)
    total_tfsa += d_fin["tfsa_cvplp"]
    d_fin["tfsa_isf"] = st.number_input("Amount in Individual segregated funds", min_value=0.0, value=0.0, step=500.0,
                                            key="tfsa_isf_"+which)
    total_tfsa += d_fin["tfsa_isf"]
    d_fin["tfsa_etf"] = st.number_input("Amount in ETFs", min_value=0.0, value=0.0, step=500.0, key="tfsa_etf_"+which)
    total_tfsa += d_fin["tfsa_etf"]
    if total_tfsa != d_fin["bal_tfsa"]:
        st.error("Total amount in financial products ({} $) is not equal to amount in financial account ({} $)".format(total_tfsa, d_fin["bal_tfsa"]))
        st.stop()
    else:
        st.success("Total amount in financial products ({} $) = amount in financial account ({} $)".format(total_tfsa, d_fin["bal_tfsa"]))

    st.markdown("### Other Reg")
    d_fin["bal_other_reg"] = st.number_input("Amount in Other Reg account", min_value=0.0, step=500.0, key="bal_other_reg_"+which)
    d_fin["cont_rate_other_reg"] = st.number_input("Fraction of your earnings you plan to save in your Other Reg account (in %)", min_value=0.00, max_value=100.00, key="cont_rate_other_reg_"+which)
    d_fin["cont_rate_other_reg"] /= 100.0
    d_fin["withdrawal_other_reg"] = st.number_input("Amount of your Other Reg account you plan to spend (in $)",
                                                        min_value=0.0, step=500.0, key="withdrawal_other_reg_"+which)
    total_bal_other_reg = 0
    d_fin["bal_other_reg_crsa"] = st.number_input("Amount in Checking or regular savings account", min_value=0.0, value=0.0,
                                            step=500.0, key="bal_other_reg_crsa_"+which)
    total_bal_other_reg += d_fin["bal_other_reg_crsa"]
    d_fin["bal_other_reg_hipsa"] = st.number_input("Amount in High interest/premium savings account", min_value=0.0, value=0.0,
                                            step=500.0, key="bal_other_reg_hipsa_"+which)
    total_bal_other_reg += d_fin["bal_other_reg_hipsa"]
    d_fin["bal_other_reg_mf"] = st.number_input("Amount in Mutual funds", min_value=0.0, value=0.0, step=500.0,
                                            key="bal_other_reg_mf_"+which)
    total_bal_other_reg += d_fin["bal_other_reg_mf"]
    d_fin["bal_other_reg_stocks"] = st.number_input("Amount in Stocks", min_value=0.0, value=0.0, step=500.0,
                                            key="bal_other_reg_stocks_"+which)
    total_bal_other_reg += d_fin["bal_other_reg_stocks"]
    d_fin["bal_other_reg_bonds"] = st.number_input("Amount in Bonds", min_value=0.0, value=0.0, step=500.0,
                                            key="bal_other_reg_bonds_"+which)
    total_bal_other_reg += d_fin["bal_other_reg_bonds"]
    d_fin["bal_other_reg_gic"] = st.number_input("Amount in GICs", min_value=0.0, value=0.0, step=500.0,
                                            key="bal_other_reg_gic_"+which)
    total_bal_other_reg += d_fin["bal_other_reg_gic"]
    d_fin["bal_other_reg_cvplp"] = st.number_input("Amount in Cash value of permanent life policy", min_value=0.0, 
                                            value=0.0, step=500.0, key="bal_other_reg_cvplp_"+which)
    total_bal_other_reg += d_fin["bal_other_reg_cvplp"]
    d_fin["bal_other_reg_isf"] = st.number_input("Amount in Individual segregated funds", min_value=0.0, value=0.0, step=500.0,
                                            key="bal_other_reg_isf_"+which)
    total_bal_other_reg += d_fin["bal_other_reg_isf"]
    d_fin["bal_other_reg_etf"] = st.number_input("Amount in ETFs", min_value=0.0, value=0.0, step=500.0,
                                            key="bal_other_reg_etf_"+which)
    total_bal_other_reg += d_fin["bal_other_reg_etf"]
    if total_bal_other_reg != d_fin["bal_other_reg"]:
        st.error("Total amount in financial products ({} $) is not equal to amount in financial account ({} $)".format(total_bal_other_reg, d_fin["bal_other_reg"]))
        st.stop()
    else:
        st.success("Total amount in financial products ({} $) = amount in financial account ({} $)".format(total_bal_other_reg, d_fin["bal_other_reg"]))

    st.markdown("### Unreg")
    d_fin["bal_unreg"] = st.number_input("Amount in Unreg account", min_value=0.0, step=500.0, key="bal_unreg_"+which)
    d_fin["cont_rate_unreg"] = st.number_input("Fraction of your earnings you plan to save in your Unreg account (in %)",
        min_value=0.00, max_value=100.00, key="cont_rate_unreg_"+which)
    d_fin["cont_rate_unreg"] /= 100.0
    d_fin["withdrawal_unreg"] = st.number_input("Amount of your Unreg account you plan to spend (in $)",
                                                    min_value=0.0, step=500.0, key="withdrawal_unreg_"+which)
    total_bal_unreg = 0
    d_fin["bal_unreg_crsa"] = st.number_input("Amount in Checking or regular savings account", min_value=0.0, value=0.0,
                                            step=500.0, key="bal_unreg_crsa_"+which)
    total_bal_unreg += d_fin["bal_unreg_crsa"]
    d_fin["bal_unreg_hipsa"] = st.number_input("Amount in High interest/premium savings account", min_value=0.0, value=0.0,
                                            step=500.0, key="bal_unreg_hipsa_"+which)
    total_bal_unreg += d_fin["bal_unreg_hipsa"]
    d_fin["bal_unreg_mf"] = st.number_input("Amount in Mutual funds", min_value=0.0, value=0.0, step=500.0,
                                            key="bal_unreg_mf_"+which)
    total_bal_unreg += d_fin["bal_unreg_mf"]
    d_fin["bal_unreg_stocks"] = st.number_input("Amount in Stocks", min_value=0.0, value=0.0, step=500.0,
                                            key="bal_unreg_stocks_"+which)
    total_bal_unreg += d_fin["bal_unreg_stocks"]
    d_fin["bal_unreg_bonds"] = st.number_input("Amount in Bonds", min_value=0.0, value=0.0, step=500.0,
                                            key="bal_unreg_bonds_"+which)
    total_bal_unreg += d_fin["bal_unreg_bonds"]
    d_fin["bal_unreg_gic"] = st.number_input("Amount in GICs", min_value=0.0, value=0.0, step=500.0,
                                            key="bal_unreg_gic_"+which)
    total_bal_unreg += d_fin["bal_unreg_gic"]
    d_fin["bal_unreg_cvplp"] = st.number_input("Amount in Cash value of permanent life policy", min_value=0.0, 
                                            value=0.0, step=500.0, key="bal_unreg_cvplp_"+which)
    total_bal_unreg += d_fin["bal_unreg_cvplp"]
    d_fin["bal_unreg_isf"] = st.number_input("Amount in Individual segregated funds", min_value=0.0, value=0.0, step=500.0,
                                            key="bal_unreg_isf_"+which)
    total_bal_unreg += d_fin["bal_unreg_isf"]
    d_fin["bal_unreg_etf"] = st.number_input("Amount in ETFs", min_value=0.0, value=0.0, step=500.0,
                                            key="bal_unreg_etf_"+which)
    total_bal_unreg += d_fin["bal_unreg_etf"]
    if total_bal_unreg != d_fin["bal_unreg"]:
        st.error("Total amount in financial products ({} $) is not equal to amount in financial account ({} $)".format(total_bal_unreg, d_fin["bal_unreg"]))
        st.stop()
    else:
        st.success("Total amount in financial products ({} $) = amount in financial account ({} $)".format(total_bal_unreg, d_fin["bal_unreg"]))

    st.markdown("### Gains and Losses in Unregistered Account")
    d_fin['cap_gains_unreg'] = st.number_input("Balance of unrealized capital gains as of December 31, 2020",
            min_value=0.0, step=500.0, key="cap_gains_unreg_"+which)
    d_fin['realized_losses_unreg'] = st.number_input("Realized losses in capital on unregistered account as of December 31, 2020",
            min_value=0.0, step=500.0, key="realized_losses_unreg_"+which)

    return d_fin

def prepare_RRI(df):
    results = main.run_simulations(df, nsim=20, n_jobs=1, non_stochastic=False,
                                   base_year=2020)
    results.merge()
    df_res = results.df_merged
    cons_floor = 0
    if len(df_res[df_res["cons_bef"] < cons_floor]) + len(df_res[df_res["cons_after"] < cons_floor]) > 0:
        st.error("Consumption is negative")
        st.stop()
    df_res['RRI'] = df_res.cons_after / df_res.cons_bef * 100
    return df_res

# graph
def plot_RRI(df, RRI_cutoff=65):
    sns.set(style="ticks")
    fig, ax = plt.subplots()
    ax = sns.kdeplot(df.RRI, color='white', bw_adjust=0.5) # kde_kws = {'shade': True, 'linewidth': 2})
    sns.despine(fig=fig, left=True)
    ax.set(xlabel='RRI Score', ylabel='')
    ax.set(yticklabels=[])
    plt.title('Probability Density Function')
    plt.yticks([])

    # compute probability of being prepared
    x, y = ax.get_lines()[0].get_data()
    n_prepared = (df.RRI > RRI_cutoff).sum()

    # threshold
    prepared = (x >= RRI_cutoff) & (x <= max(df.RRI))
    unprepared = (x <= RRI_cutoff) & (x >= min(df.RRI))
    plt.fill_between(x, y, where=prepared, color='seagreen', alpha=0.5)
    plt.fill_between(x, y, where=unprepared, color='salmon', alpha=0.5)
    plt.annotate(f'RRI > {RRI_cutoff}\nin {n_prepared}% of {len(df)} sim.',
        xy=(max(x)/2, max(y)/2), c='seagreen')
    plt.grid()
    return plt

def prepare_decomposition(df):
    results = main.run_simulations(df, nsim=1, n_jobs=1, non_stochastic=True)
    results.merge()
    df_res = results.df_merged
    return df_res

def extract_to_dict(var, new_name, d_hh, d_results):
    if f's_{var}' in d_hh:
        d_results[new_name] = d_hh[var] + d_hh[f's_{var}']
    else:
        d_results[new_name] = d_hh[var]
        
def prepare_dict_results(d_hh):
    d_variables = {'pension_after': 'pension',
                   'annuity_rrsp_after': 'annuity rrsp',
                   'annuity_rpp_dc_after': 'RPP DC',
                   'annuity_non_rrsp_after': 'annuity non-rrsp',
                   'cons_after': 'consumption',
                   'debt_payments_after': 'debt payments',
                   'fam_net_tax_liability_after': 'net tax liability',
                   'cpp_after': 'cpp',
                   'gis_after': 'gis',
                   'oas_after': 'oas',
                   'rpp_db_benefits_after': 'RPP DB'}        

    d_results = {}
    for var, name in d_variables.items():
        extract_to_dict(var, name, d_hh, d_results)
    return d_results

st.markdown(f"""<style>
    .reportview-container .main .block-container{{
    max-width: 1280px;
    padding-top: 0em;
    }}</style>""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>CPR Simulator</h1>", unsafe_allow_html=True)

col_p1, _, col_p2 = st.beta_columns([0.5, 0.1, 0.4])

with col_p1:
    d_hh = ask_hh()
    df = create_dataframe(d_hh)
    df_res = prepare_RRI(df)
    fin_acc_cols = ['bal_rrsp', 'bal_tfsa', 'bal_other_reg', 'bal_unreg', 'cont_rate_rrsp', 'cont_rate_tfsa', 'cont_rate_other_reg',
                'cont_rate_unreg', 'withdrawal_rrsp', 'withdrawal_tfsa', 'withdrawal_other_reg', 'withdrawal_unreg',
                'cap_gains_unreg', 'realized_losses_unreg', 'init_room_rrsp', 'init_room_tfsa']

    if df["couple"][0] == False:
        for i in fin_acc_cols:
            if ~df[i].notnull()[0] == 1:
                df[i].fillna(0, inplace=True)
    else:
        for i in fin_acc_cols:
            if ~df[i].notnull()[0] == 1:
                df[i].fillna(0, inplace=True)
            if ~df["s_"+i].notnull()[0] == 1:
                df["s_"+i].fillna(0, inplace=True)

if st.button("Show visualizations", False):
    df_res = prepare_RRI(df)
    with col_p2:
        st.pyplot(plot_RRI(df_res, RRI_cutoff=80))

        d_results = prepare_dict_results(df_res.fillna(0).to_dict('records')[0])

        l_income = ['oas', 'gis', 'cpp', 'pension', 'RPP DC', 'RPP DB', 'annuity rrsp', 'annuity non-rrsp']

        l_expenses = ['net tax liability', 'debt payments']

        plt.figure(figsize=(10, 8))

        income = 0
        d_income = {}
        labels = []
        for var in l_income:
            d_income[var] = income
            income += d_results[var]
        for var in reversed(l_income):
            if d_results[var] != 0:
                plt.bar(0, d_results[var], bottom=d_income[var], width=1)
                labels.append(var)

        expenses = 0
        d_expenses = {}
        for var in l_expenses:
            d_expenses[var] = expenses
            expenses += d_results[var]
        for var in l_expenses:
            if d_results[var] != 0:
                plt.bar(1, -d_results[var], bottom=-d_expenses[var], width=1)
                labels.append(var)      
        plt.bar(2, d_results['consumption'], width=1)
        labels.append('consumption')

        plt.legend(labels, loc='upper center') 
        plt.axhline(0, color='black', linestyle='--', linewidth=1)
        plt.xticks((0, 1, 2), ('income', 'expenses', 'consumption'))
        plt.grid()

        st.pyplot(plt)

        l_income = ['oas', 'gis', 'cpp', 'pension', 'RPP DC', 'RPP DB', 'annuity rrsp', 'annuity non-rrsp']

        l_expenses = ['consumption', 'debt payments', 'net tax liability']

        plt.figure(figsize=(10, 8))

        income = 0
        d_income = {}
        labels = []
        for var in l_income:
            d_income[var] = income
            income += d_results[var]
        for var in reversed(l_income):
            if d_results[var] != 0:
                plt.bar(0, d_results[var], bottom=d_income[var], width=0.5)
                labels.append(var)

        expenses = 0
        d_expenses = {}
        for var in l_expenses:
            d_expenses[var] = expenses
            expenses += d_results[var]
        for var in reversed(l_expenses):
            if d_results[var] != 0:
                plt.bar(1, d_results[var], bottom=d_expenses[var], width=0.5)
                labels.append(var) 

        plt.legend(labels, loc='upper center') 
        plt.axhline(0, color='black', linestyle='--', linewidth=1)
        plt.xticks((0, 1), ('income', 'expenses'))
        plt.grid()

        st.pyplot(plt)

    st.markdown("## Scroll to top to see the visualizations!")
    #st.balloons()