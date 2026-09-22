using System;
using System.Collections.Generic;
using ChequeVerification.Web.Models.Entities;
using Microsoft.EntityFrameworkCore;

namespace ChequeVerification.Web.Data;

public partial class ChequeVerificationDbContext : DbContext
{
    public ChequeVerificationDbContext(DbContextOptions<ChequeVerificationDbContext> options)
        : base(options)
    {
    }

    public virtual DbSet<AuditLog> AuditLogs { get; set; }

    public virtual DbSet<Cheque> Cheques { get; set; }

    public virtual DbSet<Customer> Customers { get; set; }

    public virtual DbSet<ExtractedSignature> ExtractedSignatures { get; set; }

    public virtual DbSet<ReferenceSignature> ReferenceSignatures { get; set; }

    public virtual DbSet<Role> Roles { get; set; }

    public virtual DbSet<SignatureComparison> SignatureComparisons { get; set; }

    public virtual DbSet<User> Users { get; set; }

    public virtual DbSet<VerificationResult> VerificationResults { get; set; }

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        modelBuilder.Entity<AuditLog>(entity =>
        {
            entity.ToTable("AuditLog");

            entity.Property(e => e.Action).HasMaxLength(100);
            entity.Property(e => e.CreatedAt).HasDefaultValueSql("(sysdatetime())", "DF_AuditLog_CreatedAt");
            entity.Property(e => e.Description).HasMaxLength(1000);
            entity.Property(e => e.EntityName).HasMaxLength(100);

            entity.HasOne(d => d.User).WithMany(p => p.AuditLogs)
                .HasForeignKey(d => d.UserId)
                .HasConstraintName("FK_AuditLog_User");
        });

        modelBuilder.Entity<Cheque>(entity =>
        {
            entity.ToTable("Cheque");

            entity.Property(e => e.Amount).HasColumnType("decimal(18, 2)");
            entity.Property(e => e.ChequeNumber).HasMaxLength(50);
            entity.Property(e => e.ImagePath).HasMaxLength(500);
            entity.Property(e => e.Status).HasDefaultValue((byte)1, "DF_Cheque_Status");
            entity.Property(e => e.UploadedAt).HasDefaultValueSql("(sysdatetime())", "DF_Cheque_UploadedAt");

            entity.HasOne(d => d.Customer).WithMany(p => p.Cheques)
                .HasForeignKey(d => d.CustomerId)
                .OnDelete(DeleteBehavior.ClientSetNull)
                .HasConstraintName("FK_Cheque_Customer");

            entity.HasOne(d => d.ImportedByUser).WithMany(p => p.Cheques)
                .HasForeignKey(d => d.ImportedByUserId)
                .OnDelete(DeleteBehavior.ClientSetNull)
                .HasConstraintName("FK_Cheque_ImportedByUser");
        });

        modelBuilder.Entity<Customer>(entity =>
        {
            entity.ToTable("Customer");

            entity.HasIndex(e => e.CustomerNumber, "UQ_Customer_CustomerNumber").IsUnique();

            entity.Property(e => e.AccountNumber).HasMaxLength(50);
            entity.Property(e => e.CreatedAt).HasDefaultValueSql("(sysdatetime())", "DF_Customer_CreatedAt");
            entity.Property(e => e.CustomerNumber).HasMaxLength(50);
            entity.Property(e => e.Email).HasMaxLength(255);
            entity.Property(e => e.FullName).HasMaxLength(150);
            entity.Property(e => e.NationalId).HasMaxLength(50);
            entity.Property(e => e.Phone).HasMaxLength(30);
        });

        modelBuilder.Entity<ExtractedSignature>(entity =>
        {
            entity.ToTable("ExtractedSignature");

            entity.HasIndex(e => e.ChequeId, "UQ_ExtractedSignature_Cheque").IsUnique();

            entity.Property(e => e.ExtractedAt).HasDefaultValueSql("(sysdatetime())", "DF_ExtractedSignature_ExtractedAt");
            entity.Property(e => e.ExtractionConfidence).HasColumnType("decimal(5, 4)");
            entity.Property(e => e.FileHash).HasMaxLength(128);
            entity.Property(e => e.ImagePath).HasMaxLength(500);

            entity.HasOne(d => d.Cheque).WithOne(p => p.ExtractedSignature)
                .HasForeignKey<ExtractedSignature>(d => d.ChequeId)
                .OnDelete(DeleteBehavior.ClientSetNull)
                .HasConstraintName("FK_ExtractedSignature_Cheque");
        });

        modelBuilder.Entity<ReferenceSignature>(entity =>
        {
            entity.ToTable("ReferenceSignature");

            entity.Property(e => e.CreatedAt).HasDefaultValueSql("(sysdatetime())", "DF_ReferenceSignature_CreatedAt");
            entity.Property(e => e.FileHash).HasMaxLength(128);
            entity.Property(e => e.ImagePath).HasMaxLength(500);
            entity.Property(e => e.IsActive).HasDefaultValue(true, "DF_ReferenceSignature_IsActive");

            entity.HasOne(d => d.Customer).WithMany(p => p.ReferenceSignatures)
                .HasForeignKey(d => d.CustomerId)
                .OnDelete(DeleteBehavior.ClientSetNull)
                .HasConstraintName("FK_ReferenceSignature_Customer");
        });

        modelBuilder.Entity<Role>(entity =>
        {
            entity.ToTable("Role");

            entity.HasIndex(e => e.Name, "UQ_Role_Name").IsUnique();

            entity.Property(e => e.Description).HasMaxLength(255);
            entity.Property(e => e.Name).HasMaxLength(50);
        });

        modelBuilder.Entity<SignatureComparison>(entity =>
        {
            entity.HasKey(e => e.ComparisonId);

            entity.ToTable("SignatureComparison");

            entity.HasIndex(e => new { e.VerificationId, e.ReferenceSignatureId }, "UQ_SignatureComparison_ResultReference").IsUnique();

            entity.Property(e => e.SimilarityScore).HasColumnType("decimal(5, 4)");

            entity.HasOne(d => d.ExtractedSignature).WithMany(p => p.SignatureComparisons)
                .HasForeignKey(d => d.ExtractedSignatureId)
                .OnDelete(DeleteBehavior.ClientSetNull)
                .HasConstraintName("FK_SignatureComparison_ExtractedSignature");

            entity.HasOne(d => d.ReferenceSignature).WithMany(p => p.SignatureComparisons)
                .HasForeignKey(d => d.ReferenceSignatureId)
                .OnDelete(DeleteBehavior.ClientSetNull)
                .HasConstraintName("FK_SignatureComparison_ReferenceSignature");

            entity.HasOne(d => d.Verification).WithMany(p => p.SignatureComparisons)
                .HasForeignKey(d => d.VerificationId)
                .OnDelete(DeleteBehavior.ClientSetNull)
                .HasConstraintName("FK_SignatureComparison_Verification");
        });

        modelBuilder.Entity<User>(entity =>
        {
            entity.ToTable("User");

            entity.HasIndex(e => e.Email, "UQ_User_Email").IsUnique();

            entity.Property(e => e.CreatedAt).HasDefaultValueSql("(sysdatetime())", "DF_User_CreatedAt");
            entity.Property(e => e.Email).HasMaxLength(255);
            entity.Property(e => e.FullName).HasMaxLength(150);
            entity.Property(e => e.PasswordHash).HasMaxLength(500);
            entity.Property(e => e.Status).HasDefaultValue((byte)1, "DF_User_Status");

            entity.HasOne(d => d.Role).WithMany(p => p.Users)
                .HasForeignKey(d => d.RoleId)
                .OnDelete(DeleteBehavior.ClientSetNull)
                .HasConstraintName("FK_User_Role");
        });

        modelBuilder.Entity<VerificationResult>(entity =>
        {
            entity.HasKey(e => e.VerificationId);

            entity.ToTable("VerificationResult");

            entity.Property(e => e.LowerThresholdUsed).HasColumnType("decimal(5, 4)");
            entity.Property(e => e.ModelName).HasMaxLength(100);
            entity.Property(e => e.ModelVersion).HasMaxLength(50);
            entity.Property(e => e.ReviewerComment).HasMaxLength(1000);
            entity.Property(e => e.SimilarityScore).HasColumnType("decimal(5, 4)");
            entity.Property(e => e.UpperThresholdUsed).HasColumnType("decimal(5, 4)");
            entity.Property(e => e.VerifiedAt).HasDefaultValueSql("(sysdatetime())", "DF_VerificationResult_VerifiedAt");

            entity.HasOne(d => d.Cheque).WithMany(p => p.VerificationResults)
                .HasForeignKey(d => d.ChequeId)
                .OnDelete(DeleteBehavior.ClientSetNull)
                .HasConstraintName("FK_VerificationResult_Cheque");

            entity.HasOne(d => d.ReviewedByUser).WithMany(p => p.VerificationResults)
                .HasForeignKey(d => d.ReviewedByUserId)
                .HasConstraintName("FK_VerificationResult_Reviewer");
        });

        OnModelCreatingPartial(modelBuilder);
    }

    partial void OnModelCreatingPartial(ModelBuilder modelBuilder);
}
