using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace ChequeVerification.Web.Migrations
{
    /// <inheritdoc />
    public partial class AllowNegativeCosineSimilarity : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.Sql("ALTER TABLE [SignatureComparison] DROP CONSTRAINT [CK_SignatureComparison_Similarity];");
            migrationBuilder.Sql("ALTER TABLE [SignatureComparison] ADD CONSTRAINT [CK_SignatureComparison_Similarity] CHECK ([SimilarityScore] >= -1 AND [SimilarityScore] <= 1);");
            migrationBuilder.Sql("ALTER TABLE [VerificationResult] DROP CONSTRAINT [CK_VerificationResult_Similarity];");
            migrationBuilder.Sql("ALTER TABLE [VerificationResult] ADD CONSTRAINT [CK_VerificationResult_Similarity] CHECK ([SimilarityScore] >= -1 AND [SimilarityScore] <= 1);");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.Sql("ALTER TABLE [SignatureComparison] DROP CONSTRAINT [CK_SignatureComparison_Similarity];");
            migrationBuilder.Sql("ALTER TABLE [SignatureComparison] ADD CONSTRAINT [CK_SignatureComparison_Similarity] CHECK ([SimilarityScore] >= 0 AND [SimilarityScore] <= 1);");
            migrationBuilder.Sql("ALTER TABLE [VerificationResult] DROP CONSTRAINT [CK_VerificationResult_Similarity];");
            migrationBuilder.Sql("ALTER TABLE [VerificationResult] ADD CONSTRAINT [CK_VerificationResult_Similarity] CHECK ([SimilarityScore] >= 0 AND [SimilarityScore] <= 1);");
        }
    }
}
