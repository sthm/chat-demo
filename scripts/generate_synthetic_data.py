"""
Generate synthetic customer support data for RAG testing.

This script creates realistic customer support Q&A pairs with retrieved chunks
that mimic the structure of the ground truth dataset.csv file.
"""
import csv
import json
import random

# Topic templates with realistic procedural content
TOPICS = {
    "payment_processing": {
        "questions": [
            "payment processing",
            "how to make a payment",
            "payment methods available",
            "automatic payments setup",
            "payment due date changes",
        ],
        "chunks": [
            """Payment Processing: Overview
Payments can be made through multiple channels including online banking, mobile app, phone, mail, or in-person at branch locations
Payments must be received by 5:00 PM ET on the due date to be considered on-time
Minimum payment amount is calculated as either $35 or 1% of balance plus interest and fees, whichever is greater
Remember: Payments take 1-3 business days to process depending on payment method
Tell customer:
If paying online: Payment will post within 1 business day if made before 8:00 PM ET
If paying by phone: Payment will post within 1 business day
If paying by mail: Allow 5-7 business days for processing
If paying at branch: Payment posts same business day""",
            """Automatic Payment Setup
Customers can set up automatic payments in three ways: Full Balance, Minimum Payment, or Fixed Amount
Access Online Banking and navigate to Account Services > Automatic Payments
Select payment frequency: Monthly on due date or bi-weekly
Verify bank account information is correct before enabling
Automatic payments will begin on the next billing cycle after setup
Important: Customer must maintain sufficient funds in linked account
If automatic payment fails due to insufficient funds, customer will be charged returned payment fee
Customer should still monitor statements even with automatic payments enabled
Changes to automatic payment settings take effect on next billing cycle""",
            """Payment Methods Available
Online: Through chase.com or mobile app - processes within 1 business day
Phone: Call automated system or speak with representative - processes within 1 business day
Mail: Send check or money order to payment address on statement - allow 5-7 days
Branch: Visit any Chase branch location - posts same business day
Wire Transfer: For large payments or expedited processing - contact customer service
Important Notes:
Do not send cash through mail
Include account number on check memo line
Keep confirmation number for all electronic payments
Payments made after due date may incur late fees even if in transit""",
        ]
    },
    "disputes_chargebacks": {
        "questions": [
            "dispute a charge",
            "chargeback process",
            "unauthorized transaction",
            "billing error dispute",
            "merchant dispute timeline",
        ],
        "chunks": [
            """Dispute Process: Filing a Dispute
Customer has 60 days from statement date to dispute a charge
Disputes can be filed online, by phone, or in writing
Provisional credit may be issued within 10 business days depending on dispute type
Required information for dispute:
- Transaction date and amount
- Merchant name
- Reason for dispute
- Any supporting documentation
Types of disputes:
- Unauthorized transactions: Charges customer didn't make or authorize
- Billing errors: Charged wrong amount or duplicate charges
- Service disputes: Goods/services not received or not as described
Tell customer: Disputed amount will be temporarily credited while investigation is in progress
Investigation typically takes 30-90 days depending on complexity""",
            """Unauthorized Transaction Disputes
If customer reports unauthorized transactions, immediately place fraud hold on account
Ask customer:
- Do they still have physical card in possession?
- When did they last use the card?
- Do they recognize the merchant or transaction?
- Has anyone else had access to card information?
For confirmed fraud:
- Issue provisional credit within 10 business days
- Cancel current card and issue replacement
- File fraud report with appropriate authorities
- Customer is not liable for unauthorized charges
For authorized user disputes:
- Customer is responsible for authorized user charges
- Can only remove authorized user going forward
- Previous charges cannot be disputed as unauthorized
Important: Customer must sign affidavit of fraud for unauthorized transaction claims""",
            """Chargeback Rights and Timeline
Under Fair Credit Billing Act, customers have protections for billing errors
Customer must notify bank in writing within 60 days of statement showing error
Bank has 30 days to acknowledge receipt of dispute
Bank must resolve dispute within 90 days (or two billing cycles)
During investigation:
- Cannot report account as delinquent to credit bureaus for disputed amount
- Cannot charge interest on disputed amount
- Must provide written notification of investigation results
If dispute is found in customer's favor:
- Permanent credit will be issued
- No interest charged on disputed amount
If dispute is found in merchant's favor:
- Provisional credit will be reversed
- Customer will owe original amount plus any accrued interest
Customer has right to appeal decision within 10 days of notification""",
        ]
    },
    "rewards_programs": {
        "questions": [
            "rewards points redemption",
            "how to earn rewards",
            "rewards program benefits",
            "combining rewards points",
            "rewards expiration policy",
        ],
        "chunks": [
            """Rewards Program Overview
Customers earn points on eligible purchases based on their card type
Standard earning rates:
- 1 point per $1 on all purchases (base cards)
- 2 points per $1 on dining and travel (preferred cards)
- 3 points per $1 on select categories (premium cards)
- 5 points per $1 on bonus category purchases
Points post to account within 1-2 billing cycles after purchase
Bonus points from sign-up offers typically post after meeting spending requirement
Points never expire as long as account remains open and in good standing
Tell customer: Points can be redeemed for cash back, travel, gift cards, or merchandise
Minimum redemption amount is 2,500 points ($25 value)
Check account summary or rewards portal for current points balance""",
            """Rewards Redemption Options
Cash Back: Redeem as statement credit, direct deposit, or check
- Statement credit posts within 1-2 billing cycles
- Direct deposit within 5-7 business days
- Check mailed within 10-14 business days
Travel: Book through rewards portal or reimburse travel purchases
- Portal bookings at standard redemption rate
- Pay Yourself Back feature for eligible purchases
- Points may have enhanced value for travel redemption
Gift Cards: Choose from over 150 retailers
- Electronic gift cards delivered instantly
- Physical gift cards mailed within 7-10 days
- Some gift cards offer bonus redemption value
Merchandise: Shop rewards catalog
- Items shipped within 2-3 weeks
- Free shipping on all redemptions
- Extended warranty on select items
Important: Redemptions cannot be reversed once processed
Partial redemptions allowed - don't need to redeem all points at once""",
            """Combining and Transferring Points
Points can be combined between eligible cards on same account
Both cards must be enrolled in same rewards program
To combine points:
- Log in to online banking
- Navigate to Rewards section
- Select Combine Points option
- Choose source and destination cards
Points transfer instantly once confirmed
Primary cardholder can transfer points to authorized users' accounts if eligible
Points cannot be transferred:
- To different customers or accounts
- Between different rewards programs
- After account closure (must redeem within 30 days)
Family members can combine points if they are on same household account
Business and personal card points cannot be combined even if same customer
Check program terms for specific transfer and combination rules""",
        ]
    },
    "card_activation": {
        "questions": [
            "activate new card",
            "card activation methods",
            "activate replacement card",
            "activation time frame",
            "card not activating",
        ],
        "chunks": [
            """Card Activation: Getting Started
New cards must be activated before use
Activation can be completed through:
- Mobile app: Instant activation through secure login
- Online banking: Navigate to Account Services > Activate Card
- Phone: Call number on activation sticker (automated or representative)
- ATM: Insert card and enter PIN
Required information for activation:
- Last 4 digits of card number
- Card expiration date
- CVV security code
- Customer identification (SSN or account info)
Tell customer: Card activates immediately after successful verification
Old card will be automatically deactivated when new card is activated
If replacing lost/stolen card, old card cannot be reactivated for security""",
            """Activation Time Frames and Deadlines
Cards should be activated within 30 days of receipt
If not activated within 45 days, card will be automatically deactivated for security
Replacement cards for expired cards activate automatically on expiration date
Temporary cards issued at branch can be activated immediately
Virtual card numbers can be generated instantly through mobile app
Activation verification:
- Confirmation number provided after activation
- Small test charge may appear and reverse within 1-2 days
- Account status will show as Active in online banking
If activation fails:
- Verify card information is entered correctly
- Check that card hasn't been reported lost or stolen
- Contact customer service if repeated failures occur""",
            """Troubleshooting Activation Issues
Common activation problems and solutions:
Card won't activate online or via app:
- Clear browser cache and cookies
- Try different browser or device
- Verify using correct login credentials
- Check for system maintenance notifications
Automated phone activation fails:
- Ensure speaking clearly for voice recognition
- Try entering information using keypad instead
- Call during business hours to speak with representative
Error message about card already being activated:
- May have been activated by mistake
- Try making small purchase to verify status
- Check online banking for account activity
Security hold preventing activation:
- Recent fraud alert on account
- Address verification needed
- Must contact customer service to resolve
Important: Never activate card if received unexpectedly - contact fraud department immediately""",
        ]
    },
    "fraud_protection": {
        "questions": [
            "fraud monitoring",
            "suspicious activity alert",
            "fraud protection features",
            "report stolen card",
            "zero liability policy",
        ],
        "chunks": [
            """Fraud Monitoring and Detection
Account monitored 24/7 for suspicious activity
Automated systems analyze:
- Purchase patterns and locations
- Transaction amounts and frequency
- High-risk merchant categories
- International transactions
- Online purchases
Customer will receive alerts via:
- Text message (if enrolled)
- Email notification
- Mobile app push notification
- Phone call for high-risk transactions
Tell customer to respond to alerts promptly:
- Text YES if transaction is legitimate
- Text NO if transaction is fraudulent
- Call number provided in alert for questions
Unrecognized alerts should be reported immediately
Remember: Bank will never ask for full card number or PIN in alert messages""",
            """Zero Liability Protection
Customers are not responsible for unauthorized transactions
Protection applies to:
- Card-present transactions with chip or signature
- Card-not-present transactions (online, phone, mail-order)
- Contactless tap-to-pay transactions
Coverage requirements:
- Customer must report unauthorized charges promptly
- Must have taken reasonable care of card
- Must not have shared PIN or account access
Important exceptions to zero liability:
- Authorized user transactions
- Transactions after failing to report lost/stolen card
- Business accounts (different coverage terms apply)
- PIN-based transactions if PIN was compromised
To maintain protection:
- Review statements regularly
- Report discrepancies within 60 days
- Keep card secure and never share details
- Report lost/stolen cards immediately
Provisional credit issued within 10 business days while investigating fraud claim""",
            """Reporting Lost or Stolen Cards
Report lost or stolen cards immediately to prevent fraud
Reporting methods:
- Mobile app: Select card and choose Report Lost/Stolen
- Online banking: Navigate to Account Services
- Phone: Call number on back of card (available 24/7)
- Branch: Visit any location during business hours
Information needed to report:
- Last 4 digits of account number
- Verification information (SSN, DOB, etc.)
- Details of when/where card was lost
- Any suspicious transactions already noticed
After reporting:
- Card is immediately deactivated
- Replacement card ordered automatically
- Expedited shipping available (2-3 business days)
- Emergency cash advance available if needed
- Temporary virtual card number can be issued instantly
Automatic payments and subscriptions:
- Customer should update payment information
- Some merchants may auto-update card information
- Monitor for any declined recurring payments
Remember: Customer is not liable for fraudulent charges after reporting""",
        ]
    },
    "balance_transfers": {
        "questions": [
            "balance transfer",
            "balance transfer fees",
            "balance transfer promotional rate",
            "how long does balance transfer take",
            "balance transfer limits",
        ],
        "chunks": [
            """Balance Transfer: Overview
Balance transfers allow moving debt from other credit cards to Chase account
Benefits of balance transfers:
- Consolidate multiple card balances
- Take advantage of promotional APR offers
- Simplify monthly payments
- Potentially save on interest charges
Balance transfer fee: Typically 3% or 5% of transfer amount (minimum $5)
Promotional APR: Check current offers (usually 0% for 12-18 months)
Standard APR applies after promotional period ends
Available credit limit: Transfer amount plus fee cannot exceed available credit
Processing time: 7-14 business days for transfer to complete
Remember: Cannot transfer balances between Chase accounts
Only transfer balances from other financial institutions""",
            """How to Request Balance Transfer
Balance transfers can be requested:
- Online: Through online banking transfer request form
- Phone: Call customer service with account information
- Mail: Complete and send balance transfer checks provided with card
- Mobile app: Use balance transfer feature in app
Required information for transfer:
- Account number of card to be paid off
- Name on account to be paid off
- Amount to transfer (full balance or partial)
- Account holder address for verification
Tell customer:
Promotional APR applies from date of transfer posting
Transfer request submitted is not a guarantee of approval
Subject to available credit limit and account standing
Transfers may be declined if:
- Insufficient available credit
- Account is past due or overlimit
- Exceeds maximum transfer limits
Check offer terms for any restrictions on transfer amounts or accounts""",
            """Balance Transfer Terms and Conditions
Important terms to understand:
Promotional APR period: Starts when transfer posts to account
- Make minimum payments on time to keep promotional rate
- Missing payment may void promotional APR
- Standard APR applies after promotional period
Balance transfer fee: Charged when transfer posts
- Included in total balance owed
- Cannot be waived or refunded
- Different offers may have different fee structures
Payments allocation: Payments generally applied to:
- Balances with highest APR first (after minimum payment)
- Check specific card terms for payment allocation rules
- Promotional balance typically paid last
Credit limit impact: Transfer amount + fee reduces available credit
- Cannot make purchases if at or over credit limit
- Returning available credit requires paying down balance
- Regular purchases may accrue interest at standard APR
Interest charges: New purchases may accrue interest during promotional period
- No grace period on new purchases if carrying balance
- Recommended to pay off transferred balance before making new purchases""",
        ]
    },
    "credit_limit_increase": {
        "questions": [
            "credit limit increase",
            "how to request credit line increase",
            "credit limit increase denial",
            "automatic credit limit increase",
            "credit limit decrease",
        ],
        "chunks": [
            """Credit Limit Increase: Requesting an Increase
Customers can request credit limit increase through:
- Online banking: Navigate to Account Services > Credit Limit Increase
- Mobile app: Select account and choose Request Credit Increase
- Phone: Call customer service for immediate review
- Automatic: May receive pre-approved increase offers
Eligibility requirements:
- Account open for at least 6 months
- Account in good standing (no late payments)
- No recent credit limit increase (within 6 months)
- Current income information required
Factors considered for approval:
- Payment history on this and other accounts
- Current credit utilization ratio
- Debt-to-income ratio
- Recent credit inquiries
- Overall creditworthiness
Tell customer: Hard credit inquiry may be required for request
Decision typically provided immediately for online requests
Some requests may require additional review (2-7 business days)""",
            """Understanding Credit Limit Decisions
Credit limit increase approved:
- New limit effective immediately
- No fee for requesting or receiving increase
- Updated limit shows in online banking and next statement
- May help improve credit utilization ratio
Credit limit increase denied:
- Written explanation sent within 7-10 business days
- Common reasons for denial:
  * Insufficient income
  * Recent late payments
  * High debt-to-income ratio
  * Too many recent credit inquiries
  * Limited credit history
- Can reapply after 6 months
- Work on improving credit profile before reapplying
Partial increase offered:
- May receive lower increase than requested
- Can accept partial increase or decline
- No penalty for declining offer
Important: Multiple credit limit requests in short period may negatively impact credit score""",
            """Automatic Credit Limit Increases and Decreases
Automatic increases:
- Bank periodically reviews accounts for increase eligibility
- May receive increase without requesting if:
  * Account in good standing for extended period
  * Consistent on-time payment history
  * Income information on file supports increase
- No hard credit inquiry for automatic increases
- Customer can decline automatic increase if desired
Credit limit decreases:
- Bank may reduce limit based on:
  * Payment delinquency
  * Decreased credit score
  * Inactivity on account
  * Economic factors
- Customer notified of decrease before effective date
- Decrease cannot put account over limit
- Can request reconsideration if decrease seems error
Managing credit limit:
- Keep credit utilization under 30% for best credit score impact
- Make on-time payments consistently
- Update income information when it increases
- Use account regularly (but responsibly)
Remember: Higher credit limit doesn't mean you should increase spending""",
        ]
    },
    "statement_inquiries": {
        "questions": [
            "view statements online",
            "paper statement request",
            "statement due date",
            "statement cycle explained",
            "missing statement",
        ],
        "chunks": [
            """Accessing Account Statements
Statements available through multiple channels:
Online Banking:
- Log in to chase.com
- Navigate to Account > Statements
- View up to 7 years of statements
- Download PDF format for records
Mobile App:
- Open app and select account
- Tap Statements & Activity
- View or download current and past statements
Paper Statements:
- Delivered by mail within 3-5 business days after cycle closes
- Can request paper statements in Account Preferences
- May incur monthly fee depending on account type
Tell customer:
Electronic statements available 24 hours after cycle closes
Paper statements mailed 2-3 days after cycle closes
Keep statements for at least 3 years for tax purposes
Download and save statements before account closure""",
            """Understanding Statement Cycle and Due Date
Statement cycle: Period of time covered by statement (typically 28-31 days)
- Cycle start date: Day after previous cycle ended
- Cycle end date: Last day included in statement
- Statement date: Date statement is generated
Payment due date: Typically 21-25 days after statement date
- Must be received by 5:00 PM ET to be considered on-time
- Due date is same day each month (may vary if due date falls on weekend/holiday)
Important dates on statement:
- Statement closing date: Last day of billing cycle
- Payment due date: Date payment must be received
- New balance: Total amount owed as of closing date
- Minimum payment due: Minimum amount to avoid late fee
Grace period: Time between cycle end and due date
- No interest on new purchases if paid in full by due date
- Only applies if previous balance was paid in full
- Carrying balance eliminates grace period""",
            """Statement Questions and Issues
Common statement inquiries:
Didn't receive statement:
- Check spam/junk folder for electronic statements
- Verify current email and mailing address in profile
- Request duplicate statement through online banking
- Contact customer service if statements consistently missing
Statement shows incorrect information:
- Review all transactions carefully
- Report discrepancies within 60 days
- Dispute errors immediately to protect rights
- Keep documentation supporting your claim
Want to change statement delivery:
- Switch between paper and electronic in Account Preferences
- Changes take effect next billing cycle
- May take 1-2 cycles to fully transition
- Paperless enrollment may include fee waiver or rewards
Changing due date:
- Can request due date change once per year
- New due date takes effect in 1-2 billing cycles
- Choose date that aligns with income/payment schedule
- Cannot change due date if account is past due
Remember: Always review statements promptly for unauthorized transactions""",
        ]
    },
    "interest_fees": {
        "questions": [
            "interest rate on my account",
            "how is interest calculated",
            "APR explained",
            "late payment fee",
            "fee waiver request",
        ],
        "chunks": [
            """Understanding Interest Rates and APR
APR (Annual Percentage Rate): Yearly cost of borrowing including interest and fees
Types of APRs on credit cards:
- Purchase APR: Rate on regular purchases
- Balance transfer APR: Rate on transferred balances
- Cash advance APR: Rate on cash advances (typically higher)
- Penalty APR: Higher rate applied for late payments
Variable vs Fixed APR:
- Variable APR: Changes based on Prime Rate
- Fixed APR: Rate stays same but can be changed with notice
- Most credit cards have variable APR
Current APR shown on:
- Monthly statement (top section)
- Online banking account summary
- Card agreement terms and conditions
Tell customer: Interest charges can be avoided by paying full balance by due date
Interest only charged on carried balances, not paid-in-full amounts
Making only minimum payment results in significant interest charges over time""",
            """How Credit Card Interest is Calculated
Interest calculation method: Average Daily Balance
Daily balance calculation:
- Add all daily balances in billing cycle
- Divide by number of days in cycle
- Multiply by daily periodic rate (APR ÷ 365)
- Result is interest charge for that cycle
Example calculation:
- Balance: $1,000
- APR: 18.99%
- Daily rate: 18.99% ÷ 365 = 0.052%
- Days in cycle: 30
- Interest charge: $1,000 × 0.052% × 30 = $15.60
Important factors affecting interest:
- When payments are made (earlier is better)
- New purchases add to daily balance
- Cash advances accrue interest immediately (no grace period)
- Different balances may have different APRs
Avoiding interest charges:
- Pay full statement balance by due date
- Pay more than minimum payment
- Avoid cash advances when possible
- Take advantage of 0% promotional offers""",
            """Common Fees and Fee Waivers
Standard credit card fees:
Late payment fee:
- $29 for first late payment
- $40 for subsequent late payments within 6 months
- Charged when minimum payment not received by due date
- Can negatively impact credit score
Returned payment fee:
- $29 if payment returned due to insufficient funds
- May also incur late payment fee
- Check with bank before making payment if concerned about funds
Over-limit fee:
- Most cards no longer charge over-limit fees
- Transactions may be declined if over limit
- Must opt-in to allow over-limit transactions
Foreign transaction fee:
- 3% of transaction amount for purchases made outside US
- Applies to online purchases from foreign merchants
- Some cards have no foreign transaction fee
Requesting fee waiver:
- Can request one-time courtesy waiver for late/returned payment fee
- More likely granted if:
  * First time occurrence
  * Long history of on-time payments
  * Account in good standing
- Call customer service to request
- Not guaranteed but often approved for good customers
Balance transfer and cash advance fees typically cannot be waived""",
        ]
    },
    "lost_stolen_cards": {
        "questions": [
            "report lost card",
            "stolen card what to do",
            "freeze my card",
            "temporary card lock",
            "emergency card replacement",
        ],
        "chunks": [
            """Reporting Lost or Stolen Cards Immediately
Time is critical when card is lost or stolen
Report immediately to prevent unauthorized charges:
Online: Log in to online banking and report card lost/stolen
Mobile app: Select card and choose Lock/Report Lost
Phone: Call number on back of card (available 24/7)
- If no card access, call main customer service line
- Have account information ready for verification
What happens when you report:
- Card is immediately deactivated
- Cannot be used for any transactions
- Automatic fraud monitoring activated
- New card ordered and shipped
- Account reviewed for suspicious activity
Tell customer: You are not liable for fraudulent charges after reporting
Report as soon as you notice card is missing
Don't wait to see if fraudulent charges occur
Quick reporting minimizes potential fraud and hassle""",
            """Temporary Card Lock vs Permanent Deactivation
Temporary card lock: Use when card is misplaced but might be found
- Available through mobile app or online banking
- Locks card instantly, preventing all transactions
- Can unlock immediately when card is located
- No new card issued while locked
- Lock duration: Until manually unlocked
- Automatic payments and subscriptions still process
- Perfect for: Misplaced card at home, temporary travel safety
Permanent deactivation: Use when card is truly lost or stolen
- Card permanently cancelled, cannot be reactivated
- New card with new number issued automatically
- Automatic payments will need updating
- Replacement card arrives in 5-7 business days
- Expedited shipping available (2-3 business days)
- Virtual card number can be issued immediately via app
Important differences:
- Locked card can be unlocked and used again
- Deactivated card requires new physical card
- Choose lock if unsure of card location
- Choose deactivation if card security compromised""",
            """Emergency Card Replacement and Temporary Solutions
When you need immediate access to funds after reporting lost/stolen card:
Emergency card replacement:
- Expedited shipping: 2-3 business days ($15 fee)
- Standard shipping: 5-7 business days (free)
- Request during initial report or call back later
- Shipped to address on file
Virtual card number:
- Generated instantly through mobile app
- Use for online purchases immediately
- Same credit limit as physical card
- Valid until physical card arrives
Emergency cash advance:
- Available at any Chase branch
- Up to available cash advance limit
- Subject to cash advance fees and APR
- Requires valid ID and account verification
Temporary card at branch:
- Available at select branch locations
- Can be used immediately at most merchants
- May have limited daily transaction amounts
- Full card will still be mailed
International emergency: Contact Chase international collect line
- Available 24/7 from anywhere in world
- Emergency card delivered to international address
- May take 3-5 business days internationally
Remember: Update all automatic payments and subscriptions with new card number""",
        ]
    }
}

def generate_answer_from_chunks(question_topic, chunks):
    """Generate a realistic synthesized answer from chunks"""
    # Extract key points from chunks
    if "payment" in question_topic.lower():
        return "Payments can be made through online banking, mobile app, phone, mail, or branch locations and must be received by 5:00 PM ET on the due date to be considered on-time. Customers can set up automatic payments for full balance, minimum payment, or fixed amount. Payment processing times vary: online and phone payments post within 1 business day, mail payments take 5-7 days, and branch payments post same business day. Automatic payments require maintaining sufficient funds in the linked account to avoid returned payment fees."
    elif "dispute" in question_topic.lower() or "chargeback" in question_topic.lower():
        return "Customers have 60 days from the statement date to dispute a charge and can file disputes online, by phone, or in writing. Provisional credit may be issued within 10 business days depending on the dispute type. Disputes can be for unauthorized transactions, billing errors, or service issues. The investigation typically takes 30-90 days, and during this time the disputed amount cannot be reported as delinquent to credit bureaus. Under the Fair Credit Billing Act, customers have specific protections for billing errors and the bank must resolve disputes within 90 days or two billing cycles."
    elif "reward" in question_topic.lower():
        return "Rewards points are earned on eligible purchases based on card type, with rates ranging from 1 to 5 points per dollar spent. Points can be redeemed for cash back (as statement credit, direct deposit, or check), travel bookings, gift cards, or merchandise. Points never expire as long as the account remains open and in good standing, with a minimum redemption amount of 2,500 points. Points can be combined between eligible cards on the same account and transferred instantly once confirmed, though points cannot be transferred between different customers or after account closure."
    elif "activat" in question_topic.lower():
        return "New cards must be activated before use and can be activated through the mobile app, online banking, phone, or ATM. Cards should be activated within 30 days of receipt and will be automatically deactivated after 45 days if not activated. Activation is instant after successful verification and the old card will be automatically deactivated when a new card is activated. Common activation issues include incorrect card information entry, browser cache problems, or security holds that may require contacting customer service."
    elif "fraud" in question_topic.lower():
        return "Accounts are monitored 24/7 for suspicious activity with automated systems analyzing purchase patterns, locations, amounts, and high-risk merchants. Customers receive alerts via text, email, mobile app, or phone call for high-risk transactions. Under zero liability protection, customers are not responsible for unauthorized transactions if they report unauthorized charges promptly and have taken reasonable care of their card. Lost or stolen cards should be reported immediately through the mobile app, online banking, phone, or branch to prevent fraud, and provisional credit is issued within 10 business days."
    elif "balance transfer" in question_topic.lower():
        return "Balance transfers allow moving debt from other credit cards to take advantage of promotional APR offers and consolidate multiple card balances. There is typically a 3% or 5% balance transfer fee (minimum $5) and the transfer amount plus fee cannot exceed available credit. Balance transfers take 7-14 business days to complete and can be requested online, by phone, or through the mobile app. Promotional APR applies from the date of transfer posting and making minimum payments on time is required to keep the promotional rate."
    elif "credit limit" in question_topic.lower():
        return "Credit limit increases can be requested through online banking, mobile app, or phone after the account has been open for at least 6 months with no late payments. Factors considered include payment history, credit utilization ratio, debt-to-income ratio, and overall creditworthiness. The bank may also automatically increase limits periodically for accounts in good standing without requiring a hard credit inquiry. Credit limits can also be decreased based on payment delinquency, decreased credit score, or account inactivity, with customers receiving notification before the decrease takes effect."
    elif "statement" in question_topic.lower():
        return "Statements are available through online banking (up to 7 years), mobile app, or delivered by mail within 3-5 business days after the cycle closes. Electronic statements are available 24 hours after the cycle closes and can be downloaded in PDF format. The statement cycle typically covers 28-31 days and the payment due date is 21-25 days after the statement date. Customers can switch between paper and electronic statements in account preferences and can request a due date change once per year."
    elif "interest" in question_topic.lower() or "apr" in question_topic.lower() or "fee" in question_topic.lower():
        return "APR (Annual Percentage Rate) represents the yearly cost of borrowing and includes purchase APR, balance transfer APR, cash advance APR, and penalty APR. Interest is calculated using the average daily balance method, where daily balances are added throughout the billing cycle, divided by the number of days, and multiplied by the daily periodic rate. Common fees include late payment fees ($29-$40), returned payment fees ($29), and foreign transaction fees (3%). Customers can request one-time courtesy fee waivers for late or returned payment fees, especially if it's a first occurrence and the account has a history of on-time payments."
    elif "lost" in question_topic.lower() or "stolen" in question_topic.lower():
        return "Lost or stolen cards should be reported immediately through online banking, mobile app, or phone (available 24/7) to prevent unauthorized charges. A temporary card lock can be used when a card is misplaced but might be found, while permanent deactivation is used when the card is truly lost or stolen. When a card is permanently deactivated, a new card is issued automatically with expedited shipping available (2-3 business days), and virtual card numbers can be generated instantly through the mobile app. Customers are not liable for fraudulent charges after reporting and emergency cash advances are available at any branch location."
    return "Information about this topic can be found in the customer support knowledge base."

def create_cited_chunks(chunks):
    """Create cited chunks array matching format of ground truth"""
    # Return 1-2 chunks as cited
    num_cited = random.choice([1, 2, min(len(chunks), 3)])
    cited = random.sample(chunks, num_cited)
    return json.dumps(cited)

def generate_synthetic_data():
    """Generate synthetic dataset.

    Structure similar to ground truth:
    - Fewer main topics (topic_key becomes the "question")
    - Multiple rows per topic (one row per chunk)
    - Each row represents a specific procedure/chunk
    """
    rows = []

    for topic_key, topic_data in TOPICS.items():
        # Use topic_key as the main "question" (like "payment_processing")
        # Convert to human-readable format
        topic_name = topic_key.replace("_", " ")

        chunks = topic_data["chunks"]

        # Create one row per chunk (like ground truth structure)
        for chunk in chunks:
            # Generate answer based on this specific chunk
            answer = generate_answer_from_chunks(topic_name, [chunk])

            # Create cited chunks - just this single chunk
            cited_chunks = json.dumps([chunk])

            rows.append({
                "question": topic_name,
                "retrieved_chunks": chunk,
                "answer": answer,
                "cited_chunks": cited_chunks
            })

    return rows

def main():
    """Generate and save synthetic data"""
    print("Generating synthetic customer support data...")

    data = generate_synthetic_data()

    output_path = "/Users/robertxu/Desktop/Projects/growth/chatbot/data/synthetic_dataset.csv"

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["question", "retrieved_chunks", "answer", "cited_chunks"])
        writer.writeheader()
        writer.writerows(data)

    print(f"✓ Generated {len(data)} synthetic examples")
    print(f"✓ Saved to: {output_path}")
    print(f"✓ Topics covered: {len(TOPICS)}")
    print(f"  - {', '.join(TOPICS.keys())}")

if __name__ == "__main__":
    main()
