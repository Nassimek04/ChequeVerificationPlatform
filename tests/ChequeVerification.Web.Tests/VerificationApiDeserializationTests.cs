using System.Text.Json;
using ChequeVerification.Web.Dtos.VerificationApi;

namespace ChequeVerification.Web.Tests;

public class VerificationApiDeserializationTests
{
    private static readonly JsonSerializerOptions Options = new(JsonSerializerDefaults.Web);

    [Fact]
    public void SignatureExtractionResponse_SnakeCaseFastApiJson_MapsToDto()
    {
        const string json = """
            {
              "success": true,
              "original_width": 800,
              "original_height": 355,
              "candidate_roi": {
                "x": 400,
                "y": 195,
                "width": 384,
                "height": 142
              },
              "signature_bbox": {
                "x": 10,
                "y": 5,
                "width": 300,
                "height": 90
              },
              "extraction_quality": 0.25,
              "image_format": "png",
              "signature_image_base64": "TEST"
            }
            """;

        var dto = JsonSerializer.Deserialize<SignatureExtractionResponseDto>(json, Options);

        Assert.NotNull(dto);
        Assert.True(dto.Success);
        Assert.Equal(800, dto.OriginalWidth);
        Assert.Equal(355, dto.OriginalHeight);
        Assert.NotNull(dto.CandidateRoi);
        Assert.Equal(400, dto.CandidateRoi.X);
        Assert.Equal(195, dto.CandidateRoi.Y);
        Assert.Equal(384, dto.CandidateRoi.Width);
        Assert.Equal(142, dto.CandidateRoi.Height);
        Assert.NotNull(dto.SignatureBbox);
        Assert.Equal(300, dto.SignatureBbox.Width);
        Assert.Equal(90, dto.SignatureBbox.Height);
        Assert.Equal(0.25, dto.ExtractionQuality);
        Assert.Equal("png", dto.ImageFormat);
        Assert.Equal("TEST", dto.SignatureImageBase64);
    }

    [Fact]
    public void SignatureDebugResponse_SnakeCaseFastApiJson_MapsToDto()
    {
        const string json = """
            {
              "success": true,
              "original_width": 800,
              "original_height": 355,
              "candidate_roi": {
                "x": 400,
                "y": 195,
                "width": 384,
                "height": 142
              },
              "signature_bbox": {
                "x": 10,
                "y": 5,
                "width": 300,
                "height": 90
              },
              "extraction_quality": 0.25,
              "message": null,
              "image_format": "png",
              "original_with_roi_base64": "AAA",
              "roi_image_base64": "BBB",
              "signature_image_base64": "CCC"
            }
            """;

        var dto = JsonSerializer.Deserialize<SignatureDebugResponseDto>(json, Options);

        Assert.NotNull(dto);
        Assert.True(dto.Success);
        Assert.Equal(800, dto.OriginalWidth);
        Assert.Equal(355, dto.OriginalHeight);
        Assert.NotNull(dto.CandidateRoi);
        Assert.Equal(384, dto.CandidateRoi.Width);
        Assert.Equal(142, dto.CandidateRoi.Height);
        Assert.NotNull(dto.SignatureBbox);
        Assert.Equal(300, dto.SignatureBbox.Width);
        Assert.Equal(90, dto.SignatureBbox.Height);
        Assert.Equal(0.25, dto.ExtractionQuality);
        Assert.Null(dto.Message);
        Assert.Equal("png", dto.ImageFormat);
        Assert.Equal("AAA", dto.OriginalWithRoiBase64);
        Assert.Equal("BBB", dto.RoiImageBase64);
        Assert.Equal("CCC", dto.SignatureImageBase64);
    }

    [Fact]
    public void SignatureDebugResponse_NoContentCase_MapsNullBboxAndQualityZero()
    {
        const string json = """
            {
              "success": true,
              "original_width": 800,
              "original_height": 355,
              "candidate_roi": {
                "x": 400,
                "y": 195,
                "width": 384,
                "height": 142
              },
              "signature_bbox": null,
              "extraction_quality": 0.0,
              "message": "Aucun contenu exploitable détecté dans la ROI candidate.",
              "image_format": "png",
              "original_with_roi_base64": "AAA",
              "roi_image_base64": "BBB",
              "signature_image_base64": null
            }
            """;

        var dto = JsonSerializer.Deserialize<SignatureDebugResponseDto>(json, Options);

        Assert.NotNull(dto);
        Assert.True(dto.Success);
        Assert.Null(dto.SignatureBbox);
        Assert.Equal(0.0, dto.ExtractionQuality);
        Assert.Equal("Aucun contenu exploitable détecté dans la ROI candidate.", dto.Message);
        Assert.Null(dto.SignatureImageBase64);
        Assert.Equal("AAA", dto.OriginalWithRoiBase64);
        Assert.Equal("BBB", dto.RoiImageBase64);
    }

    [Fact]
    public void SignatureDebugResponse_V22FullJson_GroupsAndCounters_MapsToDto()
    {
        const string json = """
            {
              "success": true,
              "extraction_pipeline_version": "2.2",
              "original_width": 800,
              "original_height": 355,
              "candidate_roi": {
                "x": 320,
                "y": 142,
                "width": 464,
                "height": 206
              },
              "analysis_zone": {
                "x": 0,
                "y": 0,
                "width": 464,
                "height": 206
              },
              "micr_band": {
                "x": 0,
                "y": 185,
                "width": 464,
                "height": 21
              },
              "signature_bbox": {
                "x": 198,
                "y": 102,
                "width": 180,
                "height": 95
              },
              "total_component_count": 42,
              "retained_component_count": 36,
              "rejected_component_count": 6,
              "micr_rejected_count": 6,
              "group_count": 2,
              "groups": [
                {
                  "index": 0,
                  "component_count": 34,
                  "bbox": {
                    "x": 8,
                    "y": 8,
                    "width": 175,
                    "height": 52
                  },
                  "ink_area": 5200,
                  "score": 0.0079,
                  "selected": false
                },
                {
                  "index": 1,
                  "component_count": 2,
                  "bbox": {
                    "x": 198,
                    "y": 102,
                    "width": 180,
                    "height": 95
                  },
                  "ink_area": 12985,
                  "score": 0.0476,
                  "selected": true
                }
              ],
              "selected_group_index": 1,
              "selection_reason": "Groupe 1 sélectionné : score de ressemblance signature 0.04757 (...)",
              "rejected_component_reasons": {
                "micr_texte_imprime": 6
              },
              "component_count": 36,
              "extraction_quality": 0.6575,
              "message": null,
              "image_format": "png",
              "original_with_roi_base64": "AAA",
              "roi_image_base64": "BBB",
              "mask_image_base64": "CCC",
              "components_all_base64": "DDD",
              "components_rejected_base64": "EEE",
              "components_image_base64": "FFF",
              "group_image_base64": "GGG",
              "groups_image_base64": "HHH",
              "signature_image_base64": "III"
            }
            """;

        var dto = JsonSerializer.Deserialize<SignatureDebugResponseDto>(json, Options);

        Assert.NotNull(dto);
        Assert.True(dto.Success);
        Assert.Equal("2.2", dto.ExtractionPipelineVersion);
        Assert.Equal(800, dto.OriginalWidth);
        Assert.Equal(355, dto.OriginalHeight);

        Assert.NotNull(dto.CandidateRoi);
        Assert.Equal(320, dto.CandidateRoi.X);
        Assert.Equal(142, dto.CandidateRoi.Y);
        Assert.Equal(464, dto.CandidateRoi.Width);
        Assert.Equal(206, dto.CandidateRoi.Height);

        Assert.NotNull(dto.AnalysisZone);
        Assert.Equal(464, dto.AnalysisZone.Width);
        Assert.Equal(206, dto.AnalysisZone.Height);

        Assert.NotNull(dto.MicrBand);
        Assert.Equal(0, dto.MicrBand.X);
        Assert.Equal(185, dto.MicrBand.Y);
        Assert.Equal(464, dto.MicrBand.Width);
        Assert.Equal(21, dto.MicrBand.Height);

        Assert.Equal(42, dto.TotalComponentCount);
        Assert.Equal(36, dto.RetainedComponentCount);
        Assert.Equal(6, dto.RejectedComponentCount);
        Assert.Equal(6, dto.MicrRejectedCount);
        Assert.Equal(2, dto.GroupCount);
        Assert.Equal(36, dto.ComponentCount);

        Assert.Equal(2, dto.Groups.Count);
        var first = dto.Groups[0];
        Assert.Equal(0, first.Index);
        Assert.Equal(34, first.ComponentCount);
        Assert.NotNull(first.Bbox);
        Assert.Equal(175, first.Bbox.Width);
        Assert.Equal(5200, first.InkArea);
        Assert.Equal(0.0079, first.Score);
        Assert.False(first.Selected);

        var second = dto.Groups[1];
        Assert.Equal(1, second.Index);
        Assert.Equal(2, second.ComponentCount);
        Assert.Equal(180, second.Bbox!.Width);
        Assert.Equal(12985, second.InkArea);
        Assert.Equal(0.0476, second.Score);
        Assert.True(second.Selected);

        Assert.Equal(1, dto.SelectedGroupIndex);
        Assert.Contains("Groupe 1 sélectionné", dto.SelectionReason);
        Assert.Equal(6, dto.RejectedComponentReasons["micr_texte_imprime"]);

        Assert.Equal(0.6575, dto.ExtractionQuality);
        Assert.Equal("png", dto.ImageFormat);
        Assert.Equal("AAA", dto.OriginalWithRoiBase64);
        Assert.Equal("BBB", dto.RoiImageBase64);
        Assert.Equal("CCC", dto.MaskImageBase64);
        Assert.Equal("DDD", dto.ComponentsAllBase64);
        Assert.Equal("EEE", dto.ComponentsRejectedBase64);
        Assert.Equal("FFF", dto.ComponentsImageBase64);
        Assert.Equal("GGG", dto.GroupImageBase64);
        Assert.Equal("HHH", dto.GroupsImageBase64);
        Assert.Equal("III", dto.SignatureImageBase64);
    }

    [Fact]
    public void SignatureDebugResponse_V23FullJson_RefinementFields_MapsToDto()
    {
        const string json = """
            {
              "success": true,
              "extraction_pipeline_version": "2.3",
              "original_width": 800,
              "original_height": 355,
              "candidate_roi": {
                "x": 320,
                "y": 142,
                "width": 464,
                "height": 206
              },
              "signature_bbox": {
                "x": 180,
                "y": 84,
                "width": 81,
                "height": 49
              },
              "dominant_component_bbox": {
                "x": 180,
                "y": 84,
                "width": 81,
                "height": 49
              },
              "dominant_component_ink": 3104,
              "dominant_component_score": 0.003468,
              "refined_component_count": 1,
              "core_component_indices": [0],
              "discarded_from_selected_group_indices": [1, 2],
              "refined_signature_bbox": {
                "x": 180,
                "y": 84,
                "width": 81,
                "height": 49
              },
              "refinement_reason": "Raffinement V2.3 : composante dominante (indice groupe 0, ...)",
              "extraction_quality": 0.7875,
              "message": null,
              "image_format": "png",
              "original_with_roi_base64": "AAA",
              "roi_image_base64": "BBB",
              "mask_image_base64": "CCC",
              "components_all_base64": "DDD",
              "components_rejected_base64": "EEE",
              "components_image_base64": "FFF",
              "group_image_base64": "GGG",
              "groups_image_base64": "HHH",
              "refined_group_image_base64": "JJJ",
              "signature_image_base64": "III"
            }
            """;

        var dto = JsonSerializer.Deserialize<SignatureDebugResponseDto>(json, Options);

        Assert.NotNull(dto);
        Assert.True(dto.Success);
        Assert.Equal("2.3", dto.ExtractionPipelineVersion);

        Assert.NotNull(dto.DominantComponentBbox);
        Assert.Equal(180, dto.DominantComponentBbox.X);
        Assert.Equal(84, dto.DominantComponentBbox.Y);
        Assert.Equal(81, dto.DominantComponentBbox.Width);
        Assert.Equal(49, dto.DominantComponentBbox.Height);
        Assert.Equal(3104, dto.DominantComponentInk);
        Assert.Equal(0.003468, dto.DominantComponentScore);
        Assert.Equal(1, dto.RefinedComponentCount);
        Assert.Equal(new[] { 0 }, dto.CoreComponentIndices);
        Assert.Equal(new[] { 1, 2 }, dto.DiscardedFromSelectedGroupIndices);

        Assert.NotNull(dto.RefinedSignatureBbox);
        Assert.Equal(180, dto.RefinedSignatureBbox.X);
        Assert.Equal(84, dto.RefinedSignatureBbox.Y);
        Assert.Equal(81, dto.RefinedSignatureBbox.Width);
        Assert.Equal(49, dto.RefinedSignatureBbox.Height);
        Assert.Contains("Raffinement V2.3", dto.RefinementReason);

        Assert.Equal(0.7875, dto.ExtractionQuality);
        Assert.Equal("JJJ", dto.RefinedGroupImageBase64);
        Assert.Equal("III", dto.SignatureImageBase64);
    }

    [Fact]
    public void ImageAnalysisResponse_SnakeCaseFastApiJson_MapsToDto()
    {
        const string json = """
            {
              "success": true,
              "width": 800,
              "height": 355,
              "channels": 3,
              "content_type": "image/png",
              "processing": {
                "decoded": true,
                "grayscale_ready": true
              }
            }
            """;

        var dto = JsonSerializer.Deserialize<ImageAnalysisResponseDto>(json, Options);

        Assert.NotNull(dto);
        Assert.True(dto.Success);
        Assert.Equal(800, dto.Width);
        Assert.Equal(355, dto.Height);
        Assert.Equal(3, dto.Channels);
        Assert.Equal("image/png", dto.ContentType);
        Assert.NotNull(dto.Processing);
        Assert.True(dto.Processing.Decoded);
        Assert.True(dto.Processing.GrayscaleReady);
    }

    [Fact]
    public void HealthResponse_SnakeCaseFastApiJson_MapsToDto()
    {
        const string json = """
            {
              "status": "ok",
              "service": "verification-api",
              "version": "0.1.0"
            }
            """;

        var dto = JsonSerializer.Deserialize<HealthResponseDto>(json, Options);

        Assert.NotNull(dto);
        Assert.Equal("ok", dto.Status);
        Assert.Equal("verification-api", dto.Service);
        Assert.Equal("0.1.0", dto.Version);
    }
}