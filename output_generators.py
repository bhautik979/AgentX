# ============================================================================
# PDF & EMAIL GENERATION - Output Tools
# Generate PDFs and send emails automatically
# ============================================================================

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from abc import ABC, abstractmethod
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import io

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, 
    PageBreak, Image
)
from reportlab.lib import colors

logger = logging.getLogger(__name__)


class EmailConfig:
    """Email Configuration"""
    
    SMTP_SERVER = "smtp.gmail.com"  # Gmail SMTP
    SMTP_PORT = 587
    
    # Your email account
    SENDER_EMAIL = "karanpadariya96@gmail.com"
    SENDER_PASSWORD = "K@r@n96,ggl)-"  # Use Gmail App Password
    
    # Or use other SMTP servers:
    # SMTP_SERVER = "smtp.office365.com"  # Outlook
    # SMTP_SERVER = "smtp.mail.yahoo.com"  # Yahoo


class PDFGenerator:
    """Generate PDFs for financial reports"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
    
    def setup_custom_styles(self):
        """Setup custom paragraph styles"""
        
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f4788'),
            spaceAfter=30,
            alignment=1  # Center
        ))
        
        # Section heading
        self.styles.add(ParagraphStyle(
            name='SectionHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2e5c8a'),
            spaceAfter=12,
            spaceBefore=12
        ))
        
        # Normal text
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['BodyText'],
            fontSize=11,
            leading=14
        ))
    
    def generate_budget_plan_pdf(self, user_data: Dict, budget_plan: Dict) -> bytes:
        """
        Generate PDF for personalized budget plan
        
        Args:
            user_data: User information
            budget_plan: Budget allocation plan
        
        Returns:
            PDF as bytes
        """
        
        # Create PDF buffer
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        story = []
        
        # Title
        title = Paragraph("Personalized Budget Plan", self.styles['CustomTitle'])
        story.append(title)
        story.append(Spacer(1, 0.2*inch))
        
        # User Info Section
        story.append(Paragraph("Personal Information", self.styles['SectionHeading']))
        
        user_info = [
            ["Name:", user_data.get("name", "N/A")],
            ["Monthly Income:", f"Rs {user_data.get('income', 0):,.2f}"],
            ["Currency:", user_data.get("currency", "INR")],
            ["Generated Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        ]
        
        user_table = Table(user_info, colWidths=[2*inch, 3*inch])
        user_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8f0f8')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        
        story.append(user_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Budget Allocation Section
        story.append(Paragraph("Recommended Budget Allocation", self.styles['SectionHeading']))
        
        # Create budget table
        budget_data = [["Category", "Amount (Rs)", "Percentage"]]
        total_income = user_data.get("income", 0)
        
        for category, details in budget_plan.items():
            if isinstance(details, dict):
                amount = details.get("amount", 0)
                percentage = (amount / total_income * 100) if total_income > 0 else 0
                budget_data.append([
                    category,
                    f"Rs {amount:,.2f}",
                    f"{percentage:.1f}%"
                ])
        
        budget_table = Table(budget_data, colWidths=[2.5*inch, 2*inch, 1.5*inch])
        budget_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2e5c8a')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')])
        ]))
        
        story.append(budget_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Recommendations
        story.append(Paragraph("Key Recommendations", self.styles['SectionHeading']))
        
        recommendations_text = """
        <b>1. Emergency Fund:</b> Maintain 3-6 months of expenses as emergency fund<br/><br/>
        <b>2. Essential Expenses:</b> Keep housing, food, and utilities under control<br/><br/>
        <b>3. Savings Target:</b> Aim to save at least 20% of your income<br/><br/>
        <b>4. Review Regularly:</b> Review and adjust your budget every 3 months<br/><br/>
        <b>5. Debt Management:</b> Prioritize high-interest debt repayment
        """
        
        story.append(Paragraph(recommendations_text, self.styles['CustomBody']))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
    
    def generate_risk_profile_pdf(self, user_data: Dict, risk_assessment: Dict) -> bytes:
        """
        Generate PDF for investment risk profile
        
        Args:
            user_data: User information
            risk_assessment: Risk profile assessment results
        
        Returns:
            PDF as bytes
        """
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        story = []
        
        # Title
        title = Paragraph("Investment Risk Profile Assessment", self.styles['CustomTitle'])
        story.append(title)
        story.append(Spacer(1, 0.2*inch))
        
        # Risk Profile Section
        risk_level = risk_assessment.get("risk_level", "Moderate")
        risk_score = risk_assessment.get("risk_score", 0)
        
        profile_data = [
            ["Risk Level:", risk_level],
            ["Risk Score:", f"{risk_score}/100"],
            ["Assessment Date:", datetime.now().strftime("%Y-%m-%d")],
            ["Recommendation:", risk_assessment.get("recommendation", "N/A")]
        ]
        
        profile_table = Table(profile_data, colWidths=[2.5*inch, 3*inch])
        profile_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#fff5e6')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        
        story.append(profile_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Suggested Asset Allocation
        story.append(Paragraph("Recommended Asset Allocation", self.styles['SectionHeading']))
        
        allocation = risk_assessment.get("asset_allocation", {})
        allocation_data = [["Asset Class", "Allocation", "Amount (Rs)"]]
        
        for asset_class, percentage in allocation.items():
            amount = (percentage / 100) * user_data.get("investment_amount", 0)
            allocation_data.append([
                asset_class,
                f"{percentage}%",
                f"Rs {amount:,.2f}"
            ])
        
        allocation_table = Table(allocation_data, colWidths=[2.5*inch, 1.5*inch, 2*inch])
        allocation_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2e5c8a')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')])
        ]))
        
        story.append(allocation_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Important Disclaimers
        story.append(Paragraph("Important Disclaimers", self.styles['SectionHeading']))
        
        disclaimer_text = """
        <b>This assessment is for informational purposes only.</b> It does not constitute investment advice, 
        and should not be considered as a recommendation to buy or sell any particular security. 
        Past performance is not indicative of future results. All investments carry risk, including 
        potential loss of principal. Consult with a qualified financial advisor before making any 
        investment decisions.
        """
        
        story.append(Paragraph(disclaimer_text, self.styles['CustomBody']))
        
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
    
    def generate_financial_report_pdf(self, title: str, content: str, data: Dict = None) -> bytes:
        """
        Generate generic financial report PDF
        
        Args:
            title: Report title
            content: Main content text
            data: Optional data to display in tables
        
        Returns:
            PDF as bytes
        """
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        story = []
        
        # Title
        title_para = Paragraph(title, self.styles['CustomTitle'])
        story.append(title_para)
        story.append(Spacer(1, 0.2*inch))
        
        # Content
        content_para = Paragraph(content, self.styles['CustomBody'])
        story.append(content_para)
        story.append(Spacer(1, 0.3*inch))
        
        # Data table if provided
        if data and isinstance(data, dict):
            table_data = [["Key", "Value"]]
            for key, value in data.items():
                table_data.append([str(key), str(value)])
            
            table = Table(table_data, colWidths=[3*inch, 2.5*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2e5c8a')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(table)
        
        # Footer
        story.append(Spacer(1, 0.3*inch))
        footer_text = f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        footer = Paragraph(footer_text, self.styles['Normal'])
        story.append(footer)
        
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()


class EmailSender:
    """Send emails with PDF attachments"""
    
    def __init__(self, 
                 smtp_server: str = EmailConfig.SMTP_SERVER,
                 smtp_port: int = EmailConfig.SMTP_PORT,
                 sender_email: str = EmailConfig.SENDER_EMAIL,
                 sender_password: str = EmailConfig.SENDER_PASSWORD):
        
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password
    
    async def send_email_with_attachment(self,
                                        recipient_email: str,
                                        subject: str,
                                        body: str,
                                        attachment_bytes: bytes = None,
                                        attachment_filename: str = None) -> bool:
        """
        Send email with optional PDF attachment
        
        Args:
            recipient_email: Recipient email address
            subject: Email subject
            body: Email body text
            attachment_bytes: PDF content as bytes
            attachment_filename: Filename for attachment
        
        Returns:
            Success status
        """
        
        try:
            # Create message
            message = MIMEMultipart()
            message["From"] = self.sender_email
            message["To"] = recipient_email
            message["Subject"] = subject
            
            # Add body
            message.attach(MIMEText(body, "html"))
            
            # Add attachment if provided
            if attachment_bytes and attachment_filename:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment_bytes)
                encoders.encode_base64(part)
                part.add_header(
                    "Content-Disposition",
                    f"attachment; filename= {attachment_filename}"
                )
                message.attach(part)
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(message)
            
            logger.info(f"Email sent successfully to {recipient_email}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False
    
    async def send_budget_plan_email(self,
                                     recipient_email: str,
                                     recipient_name: str,
                                     budget_data: Dict) -> bool:
        """
        Send budget plan via email with PDF
        
        Args:
            recipient_email: Recipient email
            recipient_name: Recipient name
            budget_data: Budget data including user_data and plan
        
        Returns:
            Success status
        """
        
        try:
            # Generate PDF
            pdf_generator = PDFGenerator()
            pdf_bytes = pdf_generator.generate_budget_plan_pdf(
                budget_data.get("user_data", {}),
                budget_data.get("plan", {})
            )
            
            # Compose email
            subject = f"Your Personalized Budget Plan - {datetime.now().strftime('%B %Y')}"
            
            body = f"""
            <html>
            <body>
            <h2>Hi {recipient_name},</h2>
            <p>We've generated your personalized budget plan for this month.</p>
            <p>Attached is your detailed budget report with recommendations for optimal 
            financial management.</p>
            <p><b>Key Highlights:</b></p>
            <ul>
            <li>Customized allocation based on your income</li>
            <li>Category-wise breakdown for better tracking</li>
            <li>Expert recommendations for financial growth</li>
            </ul>
            <p>Review the attached PDF and adjust your spending accordingly.</p>
            <p>Best regards,<br/>
            Financial AI Assistant</p>
            </body>
            </html>
            """
            
            return await self.send_email_with_attachment(
                recipient_email,
                subject,
                body,
                pdf_bytes,
                f"budget_plan_{datetime.now().strftime('%Y%m%d')}.pdf"
            )
        
        except Exception as e:
            logger.error(f"Failed to send budget plan email: {e}")
            return False
    
    async def send_risk_profile_email(self,
                                      recipient_email: str,
                                      recipient_name: str,
                                      risk_data: Dict) -> bool:
        """
        Send risk profile assessment via email with PDF
        
        Args:
            recipient_email: Recipient email
            recipient_name: Recipient name
            risk_data: Risk assessment data
        
        Returns:
            Success status
        """
        
        try:
            # Generate PDF
            pdf_generator = PDFGenerator()
            pdf_bytes = pdf_generator.generate_risk_profile_pdf(
                risk_data.get("user_data", {}),
                risk_data.get("assessment", {})
            )
            
            # Compose email
            subject = f"Your Investment Risk Profile Assessment"
            
            body = f"""
            <html>
            <body>
            <h2>Hi {recipient_name},</h2>
            <p>Your investment risk profile assessment is complete!</p>
            <p>Based on your responses to our questionnaire, we've generated a personalized 
            risk profile and recommended asset allocation.</p>
            <p><b>Your Risk Level:</b> {risk_data.get('assessment', {}).get('risk_level', 'Moderate')}</p>
            <p>Attached is your detailed risk profile report with:</p>
            <ul>
            <li>Comprehensive risk assessment</li>
            <li>Recommended asset allocation</li>
            <li>Investment guidelines tailored to your profile</li>
            </ul>
            <p>Please review and consult with a financial advisor for personalized guidance.</p>
            <p>Best regards,<br/>
            Investment AI Assistant</p>
            </body>
            </html>
            """
            
            return await self.send_email_with_attachment(
                recipient_email,
                subject,
                body,
                pdf_bytes,
                f"risk_profile_{datetime.now().strftime('%Y%m%d')}.pdf"
            )
        
        except Exception as e:
            logger.error(f"Failed to send risk profile email: {e}")
            return False


__all__ = ['PDFGenerator', 'EmailSender', 'EmailConfig']
