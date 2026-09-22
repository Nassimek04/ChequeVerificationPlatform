SSBI controlled test pack — signer 7
==========================================
Purpose: academic end-to-end cheque signature verification testing.

Enrollment:
- references/: 5 genuine signatures from SSBI signer/person_id 7.

Test cheques:
- genuine_cheques/: 10 cheques containing 10 distinct genuine signer-7 probe signatures.
- forged_cheques/: 10 cheques containing skilled forgeries targeting signer 7.
  The SSBI source package contains only 8 unique forged signer-7 annotations, so
  forged cases 09 and 10 reuse two forged signature samples on different cheque templates.

Important:
- Do NOT enroll any signature from genuine_cheques/ as a reference.
- Enroll only the 5 images in references/.
- Each cheque is a separate image ready for import.
- Signature pixels come from the supplied SSBI source annotations and are composited
  onto the supplied SSBI cheque templates; they are not redrawn by an image generator.
- This is an academic synthetic dataset, not a bank-grade validation corpus.
- See manifest.csv for exact provenance and SHA-256 hashes.
