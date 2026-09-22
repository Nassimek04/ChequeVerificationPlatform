using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace ChequeVerification.Web.Migrations
{
    /// <inheritdoc />
    public partial class Initial : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.CreateTable(
                name: "Customer",
                columns: table => new
                {
                    CustomerId = table.Column<int>(type: "int", nullable: false)
                        .Annotation("SqlServer:Identity", "1, 1"),
                    CustomerNumber = table.Column<string>(type: "nvarchar(50)", maxLength: 50, nullable: false),
                    FullName = table.Column<string>(type: "nvarchar(150)", maxLength: 150, nullable: false),
                    AccountNumber = table.Column<string>(type: "nvarchar(50)", maxLength: 50, nullable: false),
                    NationalId = table.Column<string>(type: "nvarchar(50)", maxLength: 50, nullable: true),
                    Phone = table.Column<string>(type: "nvarchar(30)", maxLength: 30, nullable: true),
                    Email = table.Column<string>(type: "nvarchar(255)", maxLength: 255, nullable: true),
                    CreatedAt = table.Column<DateTime>(type: "datetime2", nullable: false, defaultValueSql: "(sysdatetime())")
                        .Annotation("Relational:DefaultConstraintName", "DF_Customer_CreatedAt")
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_Customer", x => x.CustomerId);
                });

            migrationBuilder.CreateTable(
                name: "Role",
                columns: table => new
                {
                    RoleId = table.Column<int>(type: "int", nullable: false)
                        .Annotation("SqlServer:Identity", "1, 1"),
                    Name = table.Column<string>(type: "nvarchar(50)", maxLength: 50, nullable: false),
                    Description = table.Column<string>(type: "nvarchar(255)", maxLength: 255, nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_Role", x => x.RoleId);
                });

            migrationBuilder.CreateTable(
                name: "ReferenceSignature",
                columns: table => new
                {
                    ReferenceSignatureId = table.Column<int>(type: "int", nullable: false)
                        .Annotation("SqlServer:Identity", "1, 1"),
                    CustomerId = table.Column<int>(type: "int", nullable: false),
                    ImagePath = table.Column<string>(type: "nvarchar(500)", maxLength: 500, nullable: false),
                    FileHash = table.Column<string>(type: "nvarchar(128)", maxLength: 128, nullable: true),
                    CreatedAt = table.Column<DateTime>(type: "datetime2", nullable: false, defaultValueSql: "(sysdatetime())")
                        .Annotation("Relational:DefaultConstraintName", "DF_ReferenceSignature_CreatedAt"),
                    IsActive = table.Column<bool>(type: "bit", nullable: false, defaultValue: true)
                        .Annotation("Relational:DefaultConstraintName", "DF_ReferenceSignature_IsActive")
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_ReferenceSignature", x => x.ReferenceSignatureId);
                    table.ForeignKey(
                        name: "FK_ReferenceSignature_Customer",
                        column: x => x.CustomerId,
                        principalTable: "Customer",
                        principalColumn: "CustomerId");
                });

            migrationBuilder.CreateTable(
                name: "User",
                columns: table => new
                {
                    UserId = table.Column<int>(type: "int", nullable: false)
                        .Annotation("SqlServer:Identity", "1, 1"),
                    RoleId = table.Column<int>(type: "int", nullable: false),
                    FullName = table.Column<string>(type: "nvarchar(150)", maxLength: 150, nullable: false),
                    Email = table.Column<string>(type: "nvarchar(255)", maxLength: 255, nullable: false),
                    PasswordHash = table.Column<string>(type: "nvarchar(500)", maxLength: 500, nullable: false),
                    Status = table.Column<byte>(type: "tinyint", nullable: false, defaultValue: (byte)1)
                        .Annotation("Relational:DefaultConstraintName", "DF_User_Status"),
                    CreatedAt = table.Column<DateTime>(type: "datetime2", nullable: false, defaultValueSql: "(sysdatetime())")
                        .Annotation("Relational:DefaultConstraintName", "DF_User_CreatedAt"),
                    LastLogin = table.Column<DateTime>(type: "datetime2", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_User", x => x.UserId);
                    table.ForeignKey(
                        name: "FK_User_Role",
                        column: x => x.RoleId,
                        principalTable: "Role",
                        principalColumn: "RoleId");
                });

            migrationBuilder.CreateTable(
                name: "AuditLog",
                columns: table => new
                {
                    AuditLogId = table.Column<long>(type: "bigint", nullable: false)
                        .Annotation("SqlServer:Identity", "1, 1"),
                    UserId = table.Column<int>(type: "int", nullable: true),
                    Action = table.Column<string>(type: "nvarchar(100)", maxLength: 100, nullable: false),
                    EntityName = table.Column<string>(type: "nvarchar(100)", maxLength: 100, nullable: false),
                    EntityId = table.Column<int>(type: "int", nullable: true),
                    Description = table.Column<string>(type: "nvarchar(1000)", maxLength: 1000, nullable: true),
                    CreatedAt = table.Column<DateTime>(type: "datetime2", nullable: false, defaultValueSql: "(sysdatetime())")
                        .Annotation("Relational:DefaultConstraintName", "DF_AuditLog_CreatedAt")
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_AuditLog", x => x.AuditLogId);
                    table.ForeignKey(
                        name: "FK_AuditLog_User",
                        column: x => x.UserId,
                        principalTable: "User",
                        principalColumn: "UserId");
                });

            migrationBuilder.CreateTable(
                name: "Cheque",
                columns: table => new
                {
                    ChequeId = table.Column<int>(type: "int", nullable: false)
                        .Annotation("SqlServer:Identity", "1, 1"),
                    CustomerId = table.Column<int>(type: "int", nullable: false),
                    ImportedByUserId = table.Column<int>(type: "int", nullable: false),
                    ChequeNumber = table.Column<string>(type: "nvarchar(50)", maxLength: 50, nullable: false),
                    Amount = table.Column<decimal>(type: "decimal(18,2)", nullable: true),
                    IssueDate = table.Column<DateOnly>(type: "date", nullable: true),
                    ImagePath = table.Column<string>(type: "nvarchar(500)", maxLength: 500, nullable: false),
                    Status = table.Column<byte>(type: "tinyint", nullable: false, defaultValue: (byte)1)
                        .Annotation("Relational:DefaultConstraintName", "DF_Cheque_Status"),
                    UploadedAt = table.Column<DateTime>(type: "datetime2", nullable: false, defaultValueSql: "(sysdatetime())")
                        .Annotation("Relational:DefaultConstraintName", "DF_Cheque_UploadedAt")
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_Cheque", x => x.ChequeId);
                    table.ForeignKey(
                        name: "FK_Cheque_Customer",
                        column: x => x.CustomerId,
                        principalTable: "Customer",
                        principalColumn: "CustomerId");
                    table.ForeignKey(
                        name: "FK_Cheque_ImportedByUser",
                        column: x => x.ImportedByUserId,
                        principalTable: "User",
                        principalColumn: "UserId");
                });

            migrationBuilder.CreateTable(
                name: "ExtractedSignature",
                columns: table => new
                {
                    ExtractedSignatureId = table.Column<int>(type: "int", nullable: false)
                        .Annotation("SqlServer:Identity", "1, 1"),
                    ChequeId = table.Column<int>(type: "int", nullable: false),
                    ImagePath = table.Column<string>(type: "nvarchar(500)", maxLength: 500, nullable: false),
                    FileHash = table.Column<string>(type: "nvarchar(128)", maxLength: 128, nullable: true),
                    ExtractionConfidence = table.Column<decimal>(type: "decimal(5,4)", nullable: true),
                    ExtractedAt = table.Column<DateTime>(type: "datetime2", nullable: false, defaultValueSql: "(sysdatetime())")
                        .Annotation("Relational:DefaultConstraintName", "DF_ExtractedSignature_ExtractedAt")
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_ExtractedSignature", x => x.ExtractedSignatureId);
                    table.ForeignKey(
                        name: "FK_ExtractedSignature_Cheque",
                        column: x => x.ChequeId,
                        principalTable: "Cheque",
                        principalColumn: "ChequeId");
                });

            migrationBuilder.CreateTable(
                name: "VerificationResult",
                columns: table => new
                {
                    VerificationId = table.Column<int>(type: "int", nullable: false)
                        .Annotation("SqlServer:Identity", "1, 1"),
                    ChequeId = table.Column<int>(type: "int", nullable: false),
                    ReviewedByUserId = table.Column<int>(type: "int", nullable: true),
                    SimilarityScore = table.Column<decimal>(type: "decimal(5,4)", nullable: false),
                    LowerThresholdUsed = table.Column<decimal>(type: "decimal(5,4)", nullable: false),
                    UpperThresholdUsed = table.Column<decimal>(type: "decimal(5,4)", nullable: false),
                    AutomaticDecision = table.Column<byte>(type: "tinyint", nullable: false),
                    FinalDecision = table.Column<byte>(type: "tinyint", nullable: true),
                    ModelName = table.Column<string>(type: "nvarchar(100)", maxLength: 100, nullable: true),
                    ModelVersion = table.Column<string>(type: "nvarchar(50)", maxLength: 50, nullable: true),
                    VerifiedAt = table.Column<DateTime>(type: "datetime2", nullable: false, defaultValueSql: "(sysdatetime())")
                        .Annotation("Relational:DefaultConstraintName", "DF_VerificationResult_VerifiedAt"),
                    ReviewerComment = table.Column<string>(type: "nvarchar(1000)", maxLength: 1000, nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_VerificationResult", x => x.VerificationId);
                    table.ForeignKey(
                        name: "FK_VerificationResult_Cheque",
                        column: x => x.ChequeId,
                        principalTable: "Cheque",
                        principalColumn: "ChequeId");
                    table.ForeignKey(
                        name: "FK_VerificationResult_Reviewer",
                        column: x => x.ReviewedByUserId,
                        principalTable: "User",
                        principalColumn: "UserId");
                });

            migrationBuilder.CreateTable(
                name: "SignatureComparison",
                columns: table => new
                {
                    ComparisonId = table.Column<int>(type: "int", nullable: false)
                        .Annotation("SqlServer:Identity", "1, 1"),
                    VerificationId = table.Column<int>(type: "int", nullable: false),
                    ExtractedSignatureId = table.Column<int>(type: "int", nullable: false),
                    ReferenceSignatureId = table.Column<int>(type: "int", nullable: false),
                    SimilarityScore = table.Column<decimal>(type: "decimal(5,4)", nullable: false),
                    IsBestMatch = table.Column<bool>(type: "bit", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_SignatureComparison", x => x.ComparisonId);
                    table.ForeignKey(
                        name: "FK_SignatureComparison_ExtractedSignature",
                        column: x => x.ExtractedSignatureId,
                        principalTable: "ExtractedSignature",
                        principalColumn: "ExtractedSignatureId");
                    table.ForeignKey(
                        name: "FK_SignatureComparison_ReferenceSignature",
                        column: x => x.ReferenceSignatureId,
                        principalTable: "ReferenceSignature",
                        principalColumn: "ReferenceSignatureId");
                    table.ForeignKey(
                        name: "FK_SignatureComparison_Verification",
                        column: x => x.VerificationId,
                        principalTable: "VerificationResult",
                        principalColumn: "VerificationId");
                });

            migrationBuilder.CreateIndex(
                name: "IX_AuditLog_UserId",
                table: "AuditLog",
                column: "UserId");

            migrationBuilder.CreateIndex(
                name: "IX_Cheque_CustomerId",
                table: "Cheque",
                column: "CustomerId");

            migrationBuilder.CreateIndex(
                name: "IX_Cheque_ImportedByUserId",
                table: "Cheque",
                column: "ImportedByUserId");

            migrationBuilder.CreateIndex(
                name: "UQ_Customer_CustomerNumber",
                table: "Customer",
                column: "CustomerNumber",
                unique: true);

            migrationBuilder.CreateIndex(
                name: "UQ_ExtractedSignature_Cheque",
                table: "ExtractedSignature",
                column: "ChequeId",
                unique: true);

            migrationBuilder.CreateIndex(
                name: "IX_ReferenceSignature_CustomerId",
                table: "ReferenceSignature",
                column: "CustomerId");

            migrationBuilder.CreateIndex(
                name: "UQ_Role_Name",
                table: "Role",
                column: "Name",
                unique: true);

            migrationBuilder.CreateIndex(
                name: "IX_SignatureComparison_ExtractedSignatureId",
                table: "SignatureComparison",
                column: "ExtractedSignatureId");

            migrationBuilder.CreateIndex(
                name: "IX_SignatureComparison_ReferenceSignatureId",
                table: "SignatureComparison",
                column: "ReferenceSignatureId");

            migrationBuilder.CreateIndex(
                name: "UQ_SignatureComparison_ResultReference",
                table: "SignatureComparison",
                columns: new[] { "VerificationId", "ReferenceSignatureId" },
                unique: true);

            migrationBuilder.CreateIndex(
                name: "IX_User_RoleId",
                table: "User",
                column: "RoleId");

            migrationBuilder.CreateIndex(
                name: "UQ_User_Email",
                table: "User",
                column: "Email",
                unique: true);

            migrationBuilder.CreateIndex(
                name: "IX_VerificationResult_ChequeId",
                table: "VerificationResult",
                column: "ChequeId");

            migrationBuilder.CreateIndex(
                name: "IX_VerificationResult_ReviewedByUserId",
                table: "VerificationResult",
                column: "ReviewedByUserId");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "AuditLog");

            migrationBuilder.DropTable(
                name: "SignatureComparison");

            migrationBuilder.DropTable(
                name: "ExtractedSignature");

            migrationBuilder.DropTable(
                name: "ReferenceSignature");

            migrationBuilder.DropTable(
                name: "VerificationResult");

            migrationBuilder.DropTable(
                name: "Cheque");

            migrationBuilder.DropTable(
                name: "Customer");

            migrationBuilder.DropTable(
                name: "User");

            migrationBuilder.DropTable(
                name: "Role");
        }
    }
}
