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
	colccd1, colccd2 = st.beta_columns(2)
	with colccd1:
		d_debts["credit_card"] = st.number_input("Amount", step=0.5, key="debt_credit_card")
	with colccd2:
		d_debts["credit_card_payment"] = st.number_input("Current monthly payment", step=0.5, key="debt_payment_credit_card", max_value=d_debts["credit_card"])

	st.markdown("### Personal Loan")
	colpl1, colpl2 = st.beta_columns(2)
	with colpl1:
		d_debts["personal_loan"] = st.number_input("Amount", step=0.5, key="debt_personal_loan")
	with colpl2:
		d_debts["personal_loan_payment"] = st.number_input("Current monthly payment", step=0.5, key="debt_payment_personal_loan", max_value=d_debts["personal_loan"])

	st.markdown("### Student Loan")
	colsl1, colsl2 = st.beta_columns(2)
	with colsl1:
		d_debts["student_loan"] = st.number_input("Amount", step=0.5, key="debt_student_loan")
	with colsl2:
		d_debts["student_loan_payment"] = st.number_input("Current monthly payment", step=0.5, key="debt_payment_student_loan", max_value=d_debts["student_loan"])

	st.markdown("### Car Loan")
	colcl1, colcl2 = st.beta_columns(2)
	with colcl1:
		d_debts["car_loan"] = st.number_input("Amount", step=0.5, key="debt_car_loan")
	with colcl2:
		d_debts["car_loan_payment"] = st.number_input("Current monthly payment", step=0.5, key="debt_payment_car_loan", max_value=d_debts["car_loan"])

	st.markdown("### Credit Line")
	colcline1, colcline2 = st.beta_columns(2)
	with colcline1:
		d_debts["credit_line"] = st.number_input("Amount", step=0.5, key="debt_credit_line")
	with colcline2:
		d_debts["credit_line_payment"] = st.number_input("Current monthly payment", step=0.5, key="debt_payment_credit_line", max_value=d_debts["credit_line"])

	st.markdown("### Other debt")
	colod1, colod2 = st.beta_columns(2)
	with colod1:
		d_debts["other_debt"] = st.number_input("Amount", step=0.5, key="debt_other_debt")
	with colod2:
		d_debts["other_debt_payment"] = st.number_input("Current monthly payment", step=0.5, key="debt_payment_other_debt", max_value=d_debts["other_debt"])

	return d_debts

def info_residence(which):
	d_res = {}
	res = "the "+which+" residence"
	colir1, colir2 = st.beta_columns(2)
	colir3, colir4 = st.beta_columns(2)
	with colir1:
		res_value_str = "Value of "+res
		d_res[f'{which}_residence'] = st.number_input(res_value_str, step=0.5, key="res_value")
	with colir2:
		res_buy_str = "Buying price of "+res
		d_res[f'price_{which}_residence'] = st.number_input(res_buy_str, step=0.5, key="res_buy")
	with colir3:
		res_mortgage_str = "Amount of mortgage on "+res
		d_res[f'{which}_mortgage'] = st.number_input(res_mortgage_str, step=0.5, key="res_mortgage")
	with colir4:
		res_mortgage_payment_str = "Monthly payment on "+res
		d_res[f'{which}_mortgage_payment'] = st.number_input(res_mortgage_payment_str, step=0.5, key="res_mortgage_payment")
	return d_res

def mix_fee():
	st.markdown("### Investments")
	df = pd.read_csv('mix_fee_assets.csv', index_col=0, usecols=range(0, 5))
	d_investments = {}
	col_mf1, col_mf2 = st.beta_columns(2)
	col_mf3, col_mf4 = st.beta_columns(2)
	col_mf5, col_mf6 = st.beta_columns(2)
	col_mf7, col_mf8 = st.beta_columns(2)
	col_mf9, col_mf10 = st.beta_columns(2)
	with col_mf1:
		st.markdown("#### Checking or regular savings account")
		d_investments["Checking or regular savings account"] = st.slider("Fraction of portfolio invested", min_value=0.0, max_value=100.0, value=10.0, step=0.5, key="mix_fee_checking")
	d_investments["Checking or regular savings account"] /= 100
	with col_mf2:
		st.markdown("#### High interest/premium savings account")
		d_investments["High interest/premium savings account"] = st.slider("Fraction of portfolio invested", min_value=0.0, max_value=100.0, value=10.0, step=0.5, key="mix_fee_premium")
	d_investments["High interest/premium savings account"] /= 100
	with col_mf3:
		st.markdown("#### Mutual funds")
		d_investments["Mutual funds"] = st.slider("Fraction of portfolio invested", min_value=0.0, max_value=100.0, value=10.0, step=0.5, key="mix_fee_mf")
	d_investments["Mutual funds"] /= 100
	with col_mf4:
		st.markdown("#### Stocks")
		d_investments["Stocks"] = st.slider("Fraction of portfolio invested", min_value=0.0, max_value=100.0, value=10.0, step=0.5, key="mix_fee_stocks")
	d_investments["Stocks"] /= 100
	with col_mf5:
		st.markdown("#### Bonds")
		d_investments["Bonds"] = st.slider("Fraction of portfolio invested", min_value=0.0, max_value=100.0, value=10.0, step=0.5, key="mix_fee_bonds")
	d_investments["Bonds"] /= 100
	with col_mf6:
		st.markdown("#### GICs")
		d_investments["GICs"] = st.slider("Fraction of portfolio invested", min_value=0.0, max_value=100.0, value=10.0, step=0.5, key="mix_fee_gic")
	d_investments["GICs"] /= 100
	with col_mf7:
		st.markdown("#### Cash value of permanent life policy")
		d_investments["Cash value of permanent life policy"] = st.slider("Fraction of portfolio invested", min_value=0.0, max_value=100.0, value=10.0, step=0.5, key="mix_fee_policy")
	d_investments["Cash value of permanent life policy"] /= 100
	with col_mf8:
		st.markdown("#### Individual segregated funds")
		d_investments["Individual segregated funds"] = st.slider("Fraction of portfolio invested", min_value=0.0, max_value=100.0, value=10.0, step=0.5, key="mix_fee_segregated")
	d_investments["Individual segregated funds"] /= 100
	with col_mf9:
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
	d_mix_fee["mix_bills"] = d_mix_fee.pop("bills")
	d_mix_fee["mix_bonds"] = d_mix_fee.pop("bonds")
	d_mix_fee["mix_equity"] = d_mix_fee.pop("equity")
	return d_mix_fee

def info_hh():
	d_others = {}
	st.markdown("### Which province do you live in?")
	d_others['prov'] = st.selectbox("Province", ["qc", "on"], key="prov")
	d_others.update(mix_fee())
	st.markdown("### Residences")
	for which in ['first', 'second']:
		which_str = "Do you own a "+which+" residence?"
		res = st.radio(which_str, ["Yes", "No"], key=which, index=1)
		if res == "Yes":
			d_others.update(info_residence(which))

	st.markdown("### Business")
	business = st.radio("Do you own a business?", ["Yes", "No"], key="business", index=1)
	colb1, colb2 = st.beta_columns(2)
	if business == "Yes":
		with colb1:
			d_others['business'] = st.number_input("Current value of the business", step=0.5, key="business_value")
		with colb2:
			d_others['price_business'] = st.number_input("Buying price of the business", step=0.5, key="business_price")

	mortgage = st.radio("Do you have any debt other than mortgage?", ["Yes", "No"], key="mortgage", index=1)
	if mortgage == "Yes":
		d_others.update(debts())

	return d_others

def fin_accounts(which):
	d_fin = {}
	st.markdown("### Savings account")

	st.markdown("### RRSP")
	colrrsp1, colrrsp2 = st.beta_columns(2)
	colrrsp3, colrrsp4 = st.beta_columns(2)
	with colrrsp1:
		d_fin["bal_rrsp"] = st.number_input("Amount in RRSP account", step=0.5, key="bal_rrsp_"+which)
	with colrrsp2:
		d_fin["cont_rate_rrsp"] = st.slider("Fraction of your earnings you plan to save in your RRSP account (in %)", min_value=0.0, max_value=100.0, step=0.5, value=10.0, key="cont_rate_rrsp_"+which)
	with colrrsp3:
		d_fin["withdrawal_rrsp"] = st.number_input("Amount of your RRSP account you plan to spend (in %)", key="withdrawal_rrsp_"+which)
	with colrrsp4:
		d_fin["init_room_rrsp"] = st.number_input("Current contribution room for RRSP", key="init_room_rrsp_"+which)

	st.markdown("### TFSA")
	coltfsa1, coltfsa2 = st.beta_columns(2)
	coltfsa3, coltfsa4 = st.beta_columns(2)
	with coltfsa1:
		d_fin["bal_tfsa"] = st.number_input("Amount in TFSA account", step=0.5, key="bal_tfsa_"+which)
	with coltfsa2:
		d_fin["cont_rate_tfsa"] = st.slider("Fraction of your earnings you plan to save in your TFSA account (in %)", min_value=0.0, max_value=100.0, step=0.5, value=10.0, key="cont_rate_tfsa_"+which)
	with coltfsa3:
		d_fin["withdrawal_tfsa"] = st.number_input("Amount of your TFSA account you plan to spend (in %)", key="withdrawal_tfsa_"+which)
	with coltfsa4:
		d_fin["init_room_tfsa"] = st.number_input("Current contribution room for TFSA", key="init_room_tfsa_"+which)

	st.markdown("### Other Reg")
	coloreg1, coloreg2 = st.beta_columns(2)
	coloreg3, coloreg4 = st.beta_columns(2)
	with coloreg1:
		d_fin["bal_other_reg"] = st.number_input("Amount in Other Reg account", step=0.5, key="bal_other_reg_"+which)
	with coloreg2:
		d_fin["cont_rate_other_reg"] = st.slider("Fraction of your earnings you plan to save in your Other Reg account (in %)", min_value=0.0, max_value=100.0, step=0.5, value=10.0, key="cont_rate_other_reg_"+which)
	with coloreg3:
		d_fin["withdrawal_other_reg"] = st.number_input("Amount of your Other Reg account you plan to spend (in %)", key="withdrawal_other_reg_"+which)

	st.markdown("### Unreg")
	colureg1, colureg2 = st.beta_columns(2)
	colureg3, colureg4 = st.beta_columns(2)
	with colureg1:
		d_fin["bal_unreg"] = st.number_input("Amount in Unreg account", step=0.5, key="bal_unreg_"+which)
	with colureg2:
		d_fin["cont_rate_unreg"] = st.slider("Fraction of your earnings you plan to save in your Unreg account (in %)", min_value=0.0, max_value=100.0, step=0.5, value=10.0, key="cont_rate_unreg_"+which)
	with colureg3:
		d_fin["withdrawal_unreg"] = st.number_input("Amount of your Unreg account you plan to spend (in %)", key="withdrawal_unreg_"+which)

	st.markdown("### Gains and Losses in Unregistered Account")
	colf1, colf2 = st.beta_columns(2)
	with colf1:
		d_fin['cap_gains_unreg'] = st.number_input("Current capital gains on unregistered account", step=0.5, key="cap_gains_unreg_"+which)
	with colf2:
		d_fin['realized_losses_unreg'] = st.number_input("Current realized losses in capital on unregistered account", step=0.5, key="realized_losses_unreg_"+which)

	return d_fin

def info_spouse(which='first'):
	d = {}
	colby, colg = st.beta_columns(2)
	with colby:
		d['byear'] = st.number_input("Birth Year", min_value=1900, max_value=2020, key="byear_"+which, value=1960)
	with colg:
		d['sex'] = st.radio("Gender", ["Male", "Female"], key="sex_"+which, index=1)
	colret, _, colcpp, _ = st.beta_columns([0.49, 0.01, 0.49, 0.01])
	with colret:
		d['ret_age'] = st.number_input("Retirement Age", min_value=2021-d['byear']+1, key="ret_age_"+which, value=65)
	with colcpp:
		d['claim_age_cpp'] = st.slider("Claim age cpp", key="claim_age_cpp_"+which, min_value=60, max_value=70, value=65)
	coled, colwage = st.beta_columns(2)
	with coled:
		d['education'] = st.selectbox("Education", ["university", "post-secondary", "high school", "less than high school"], key="education_"+which)
	with colwage:
		d['init_wage'] = st.number_input("Annual Earnings for 2018", step=1, key="init_wage_"+which, value=50000)

	colpr, colpv = st.beta_columns(2)
	with colpr:
		pension = st.radio("Do you currently receive a pension?", ["Yes", "No"], key="pension_radio_"+which, index=1)
	if pension == "Yes":
		with colpv:
			d['pension'] = st.number_input("Yearly amount of pension", step=0.5, key="pension_"+which)

	savings_plan = st.radio("Do you have any savings or plan to save in the future?", ["Yes", "No"], key="savings_plan_"+which, index=1)
	if savings_plan == "Yes":
		d.update(fin_accounts(which=which))

	# db pension
	db_pension = st.radio("Will you receive a DB pension from your current or previous employer", ["Yes", "No"], key="db_pension_"+which, index=1)
	if db_pension == "Yes":
		st.markdown("### DB Pension")
		# lower and upper bound for pension
		low_replacement_rate = min(0.02 * (d['ret_age'] - (2018 - d['byear'])), 0.70)
    	high_replacement_rate = min(0.02 * (d['ret_age'] - 18), 0.70)
    	return np.clip(row['replacement_rate_db'], low_b, high_b)
		d['replacement_rate_db'] = st.slider("Replacement rate of current DB (in %)", min_value=low_replacement_rate, max_value=high_replacement_rate,
											 step=0.5, key="replacement_rate_db_"+which, value=(low_replacement_rate - high_replacement_rate) / 2)
		d['replacement_rate_db'] /= 100
		d['rate_employee_db'] = st.slider("Contribution rate employee of current DB (in %)", min_value=0.00, max_value=9.00, step=0.5,
										  key="rate_employee_db_"+which, value=5.00)
		d['rate_employee_db'] /= 100
		d['income_previous_db'] = st.number_input("Amount of DB pension from previous employer", step=0.5, key="income_previous_db_"+which)

	# dc pension
	dc_pension = st.radio("Do you have a DC pension from current or previous employer", ["Yes", "No"], key="dc_pension_"+which, index=1)
	if dc_pension == "Yes":
		st.markdown("### DC Pension")
		d['init_dc'] = st.number_input("Current amount", step=0.5, key="init_dc_"+which)
		d['rate_employee_dc'] = st.slider("Contribution rate employee of current DC (in %)", min_value=0.00, max_value=18.00, step=0.5,
										  key="rate_employee_dc_"+which, value=5.00)
		d['rate_employee_dc'] /= 100
		d['rate_employer_dc'] = st.slider("Contribution rate  employer of current DC (in %)", min_value=0.00, max_value=18.00 - d['rate_employee-dc'],
										  step=0.5, key="rate_employer_dc_"+which, value=5.00)
		d['rate_employer_dc'] /= 100

	if which == 'second':
		d = {'s_' + k: v for k, v in d.items()}
	return d

def ask_hh():
    with st.beta_expander("FIRST SPOUSE"):
	    d_hh = info_spouse()
    spouse_ques = st.radio("Do you have a spouse?", ["Yes", "No"], index=0)
    spouse_dict = {}
    spouse_dict["Yes"] = True
    spouse_dict["No"] = False
    d_hh['couple'] =  spouse_dict[spouse_ques]
    if d_hh['couple']:
    	with st.beta_expander("SECOND SPOUSE"):
	    	d_hh.update(info_spouse('second'))

    with st.beta_expander("HOUSEHOLD"):
	    d_hh.update(info_hh())
	    d_hh['weight'] = 1
    return d_hh

def prepare_RRI(df):
	results = main.run_simulations(df, nsim=20, n_jobs=1, non_stochastic=False)
	results.merge()
	df_res = results.df_merged
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
	max-width: 900px;
	padding-top: 0em;
	}}</style>""", unsafe_allow_html=True)

d_hht = ask_hh()
dft = create_dataframe(d_hht)

if st.checkbox("Show raw data", False):
	st.dataframe(dft.transpose())

st.markdown("### Note: These visualizations are based on pre-defined dummy data!")

with open('d_hh.pkl', 'rb') as file:
    d_hh = pickle.load(file)
df = create_dataframe(d_hh)
df_res = prepare_RRI(df)
_, plt1, _ = st.beta_columns([0.25, 0.5, 0.25])
with plt1:
	st.pyplot(plot_RRI(df_res, RRI_cutoff=80))

d_results = prepare_dict_results(df_res.fillna(0).to_dict('records')[0])

l_income = ['oas', 'gis', 'cpp', 'pension', 'RPP DC', 'RPP DB', 'annuity rrsp', 'annuity non-rrsp']

l_expenses = ['net tax liability', 'debt payments']

plt_bar1, plt_bar2 = st.beta_columns(2)

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

with plt_bar1:
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

with plt_bar2:
	st.pyplot(plt)