import boto3
import logging
from botocore.exceptions import ClientError
from core.config import settings

logger = logging.getLogger(__name__)


class S3Service:
    """
    Handles all S3 operations with folder-based routing:
      approved/   — high-confidence CVs (≥ 0.75, auto-accepted)
      flagged/    — medium-confidence CVs (0.60–0.75, returned with warning)
      pending/    — low-confidence CVs (< 0.60, awaiting human review)
      corrected/  — CVs corrected by a reviewer
    Only PDF and TXT files are stored. No JSON files.
    """

    def __init__(self):
        self.bucket = settings.S3_BUCKET_NAME
        self._ready = False
        placeholder_keys = {"", "your_access_key", "your_secret_key"}

        if (
            settings.AWS_ACCESS_KEY_ID in placeholder_keys
            or settings.AWS_SECRET_ACCESS_KEY in placeholder_keys
            or not self.bucket
        ):
            logger.warning("S3Service: AWS credentials not configured — running in local-only mode.")
            self.s3 = None
            return

        try:
            self.s3 = boto3.client(
                "s3",
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.AWS_REGION,
            )
            self._ready = True
            logger.info(f"S3Service: connected to bucket '{self.bucket}' in {settings.AWS_REGION}")
        except Exception as e:
            logger.error(f"S3Service: failed to initialise boto3 client: {e}")
            self.s3 = None

    # ── Internal helpers ──────────────────────────────────────────

    def _put(self, key: str, body: bytes, content_type: str = "application/octet-stream") -> str:
        """Upload bytes to S3. Returns the S3 key on success, raises on failure."""
        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=body,
            ContentType=content_type,
            ServerSideEncryption="AES256",
        )
        logger.info(f"S3: uploaded s3://{self.bucket}/{key}")
        return key

    def _copy(self, src_key: str, dst_key: str):
        """Copy an object within the same bucket."""
        self.s3.copy_object(
            Bucket=self.bucket,
            CopySource={"Bucket": self.bucket, "Key": src_key},
            Key=dst_key,
            ServerSideEncryption="AES256",
        )
        logger.info(f"S3: copied {src_key} → {dst_key}")

    def _delete(self, key: str):
        """Delete an object from the bucket."""
        self.s3.delete_object(Bucket=self.bucket, Key=key)
        logger.info(f"S3: deleted s3://{self.bucket}/{key}")

    @staticmethod
    def _stem(filename: str) -> str:
        """Return filename without extension, e.g. 'john_doe.pdf' → 'john_doe'."""
        return filename.rsplit(".", 1)[0] if "." in filename else filename

    # ── Public API ────────────────────────────────────────────────

    def save_flagged(self, pdf_bytes: bytes, filename: str, text: str, cv_id: str) -> dict:
        """
        Save a medium-confidence (flagged) CV to flagged/.
        Confidence between INF_CONF (0.60) and HIGH_CONF (0.75).
        Stored for reference — no human review required.
        """
        if not self._ready:
            return {"status": "local-only"}

        stem = self._stem(filename)
        pdf_key = f"flagged/{cv_id}_{stem}.pdf"
        txt_key = f"flagged/{cv_id}_{stem}.txt"

        try:
            self._put(pdf_key, pdf_bytes, "application/pdf")
            self._put(txt_key, text.encode("utf-8"), "text/plain")
            return {"status": "ok", "pdf": pdf_key, "txt": txt_key}
        except ClientError as e:
            logger.error(f"S3 save_flagged failed: {e}")
            return {"status": "error", "detail": str(e)}

    def save_approved(self, pdf_bytes: bytes, filename: str, text: str, cv_id: str) -> dict:
        """
        Save a high-confidence CV directly to approved/.
        Returns dict with s3 keys, or {'status': 'local-only'} if S3 not configured.
        """
        if not self._ready:
            return {"status": "local-only"}

        stem = self._stem(filename)
        pdf_key = f"approved/{cv_id}_{stem}.pdf"
        txt_key = f"approved/{cv_id}_{stem}.txt"

        try:
            self._put(pdf_key, pdf_bytes, "application/pdf")
            self._put(txt_key, text.encode("utf-8"), "text/plain")
            return {"status": "ok", "pdf": pdf_key, "txt": txt_key}
        except ClientError as e:
            logger.error(f"S3 save_approved failed: {e}")
            return {"status": "error", "detail": str(e)}

    def save_pending(self, pdf_bytes: bytes, filename: str, text: str, cv_id: str) -> dict:
        """
        Save a low-confidence CV to pending/ until a reviewer decides.
        Returns dict with s3 keys, or {'status': 'local-only'} if S3 not configured.
        """
        if not self._ready:
            return {"status": "local-only"}

        stem = self._stem(filename)
        pdf_key = f"pending/{cv_id}_{stem}.pdf"
        txt_key = f"pending/{cv_id}_{stem}.txt"

        try:
            self._put(pdf_key, pdf_bytes, "application/pdf")
            self._put(txt_key, text.encode("utf-8"), "text/plain")
            return {"status": "ok", "pdf": pdf_key, "txt": txt_key}
        except ClientError as e:
            logger.error(f"S3 save_pending failed: {e}")
            return {"status": "error", "detail": str(e)}

    def approve_pending(self, cv_id: str, stem: str) -> dict:
        """
        Move a CV from pending/ → approved/.
        Copies both pdf and txt, then deletes originals.
        """
        if not self._ready:
            return {"status": "local-only"}

        src_pdf = f"pending/{cv_id}_{stem}.pdf"
        src_txt = f"pending/{cv_id}_{stem}.txt"
        dst_pdf = f"approved/{cv_id}_{stem}.pdf"
        dst_txt = f"approved/{cv_id}_{stem}.txt"

        try:
            self._copy(src_pdf, dst_pdf)
            self._copy(src_txt, dst_txt)
            self._delete(src_pdf)
            self._delete(src_txt)
            return {"status": "ok", "pdf": dst_pdf, "txt": dst_txt}
        except ClientError as e:
            logger.error(f"S3 approve_pending failed: {e}")
            return {"status": "error", "detail": str(e)}

    def save_corrected(
        self,
        cv_id: str,
        stem: str,
        corrected_text: str,
    ) -> dict:
        """
        Move a CV from pending/ → corrected/.
        Copies original PDF, writes a new *_corrected.txt, deletes pending files.
        """
        if not self._ready:
            return {"status": "local-only"}

        src_pdf = f"pending/{cv_id}_{stem}.pdf"
        src_txt = f"pending/{cv_id}_{stem}.txt"
        dst_pdf = f"corrected/{cv_id}_{stem}.pdf"
        dst_txt = f"corrected/{cv_id}_{stem}_corrected.txt"

        try:
            self._copy(src_pdf, dst_pdf)
            self._put(dst_txt, corrected_text.encode("utf-8"), "text/plain")
            self._delete(src_pdf)
            self._delete(src_txt)
            return {"status": "ok", "pdf": dst_pdf, "txt": dst_txt}
        except ClientError as e:
            logger.error(f"S3 save_corrected failed: {e}")
            return {"status": "error", "detail": str(e)}