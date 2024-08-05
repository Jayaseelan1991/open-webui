import logging

from fastapi import Request, UploadFile, File
from fastapi import Depends, HTTPException, status
from fastapi_sso.sso.microsoft import MicrosoftSSO

from fastapi import APIRouter
import json
from utils.mail.mail import Mail

from utils.utils import (
    get_current_user,
)

from apps.web.models.services import (
    LeaveResponse,
    LeaveForm,
)

from config import (
    CLIENT_ID,
    CLIENT_SECRET,
    TENANT,
    REDIRECT_URI,
    HR_EMAIL,
)

from constants import ERROR_MESSAGES, WEBHOOK_MESSAGES
router = APIRouter()


sso = MicrosoftSSO(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    tenant=TENANT,
    redirect_uri=REDIRECT_URI,
    allow_insecure_http=True,
    scope=["User.Read", "Directory.Read.All", "User.ReadBasic.All", "Mail.Send"],
)


############################
# Leave Application
############################


@router.post("/leave", response_model=LeaveResponse)
async def submit_leave_form(
    request: Request,form_data: LeaveForm, session_user = Depends(get_current_user)
):
    logging.info(f"Received leave form data: {form_data}")
    if session_user:
        access_token = json.loads(session_user.extra_sso)["access_token"]
        logging.info(f"SSO access_token: {access_token}")

        mail = Mail(client_id=CLIENT_ID, tenant_id=TENANT, authorization=f"Bearer {access_token}")
        
        subject = f"Leave of Absence Request - {form_data.name}"
        body = f"""
Dear HR,

I am writing to formally request a leave of absence from {form_data.leavefrom} to {form_data.leaveto}.

Thank you for considering my request. I look forward to your response.

Please see the attachment for details.

Note: This email is automatically generated by the ServiceDesk Chatbot system.

Sincerely,
{form_data.name}
"""
        recipient = HR_EMAIL
        if not recipient:
            logging.warn(f"The HR_EMAIL is None. It will be seted with userslef's email {session_user.email}")
            recipient = session_user.email
        
        logging.info(f"Preparing to send email to {recipient}")
        await mail.send_mail(subject, body, recipient, form_data)
        logging.info(f"Email sent to {recipient}")

        return {
                "email": session_user.email,
                "subject": subject,
                "recipient": recipient,
            }
    else:
        raise HTTPException(400, detail=ERROR_MESSAGES.INVALID_CRED)
