from growwapi import GrowwAPI 
import pyotp
from dotenv import load_dotenv
import os
 
load_dotenv()

api_key = os.getenv("groww_api_key")
totp_gen = pyotp.TOTP(os.getenv("groww_totp_gen")) 
totp = totp_gen.now() 
 
access_token = GrowwAPI.get_access_token(api_key, totp) 
# Use access_token to initiate GrowwAPi
growwapi = GrowwAPI(access_token)