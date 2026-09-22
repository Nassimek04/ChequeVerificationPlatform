using System.ComponentModel.DataAnnotations;

namespace ChequeVerification.Web.ViewModels.ManualReviews;

public class ManualReviewDecisionViewModel
{
    [Required]
    public int VerificationId { get; set; }

    [Required(ErrorMessage = "Veuillez choisir une décision.")]
    [Range(1, 2, ErrorMessage = "Décision invalide.")]
    public byte? FinalDecision { get; set; } // 1 = Conforme, 2 = Non conforme

    [Required(ErrorMessage = "Un commentaire est requis.")]
    [StringLength(1000, MinimumLength = 5, ErrorMessage = "Le commentaire doit contenir entre 5 et 1000 caractères.")]
    public string ReviewerComment { get; set; } = string.Empty;
}
